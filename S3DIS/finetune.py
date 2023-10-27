import random, os
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from s3dis import S3DIS, s3dis_collate_fn
from torch.utils.data import DataLoader
import sys, math
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.scheduler.cosine_lr import CosineLRScheduler
from utils.timm.optim import create_optimizer_v2
import utils.util as util
from delasemseg import DelaSemSeg
from delasemseg_mem import DelaSemSeg_Mem
from time import time
from config import s3dis_args, s3dis_warmup_args, dela_args, batch_size, learning_rate as lr, epoch, warmup, label_smoothing as ls
from tqdm import tqdm
from argparse import ArgumentParser


# torch.set_float32_matmul_precision("high")

def warmup_fn(model, dataset):
    model.train()
    traindlr = DataLoader(dataset, batch_size=len(dataset), collate_fn=s3dis_collate_fn, pin_memory=True, num_workers=6)
    for xyz, feature, indices, pts, y in traindlr:
        xyz = xyz.cuda(non_blocking=True)
        feature = feature.cuda(non_blocking=True)
        indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
        pts = pts.tolist()[::-1]
        y = y.cuda(non_blocking=True)
        with autocast():
            # p, closs = model(xyz, feature, indices, pts)
            # memory版
            p, closs, coarse_seg_loss = model(xyz, feature, indices, pts, y, 0)
            loss = F.cross_entropy(p, y) + closs + coarse_seg_loss * 0.2
        loss.backward()

parser = ArgumentParser()
parser.add_argument('--cur_id', required=True)
args = parser.parse_args()

cur_id = args.cur_id
os.makedirs(f"output/log/{cur_id}", exist_ok=True)
os.makedirs(f"output/model/{cur_id}", exist_ok=True)
logfile = f"output/log/{cur_id}/out.log"
logger = util.create_logger(logfile)

logger.info(r"base ")

train_dataset = S3DIS(s3dis_args, partition="!5", loop=30)
test_dataset = S3DIS(s3dis_args, partition="5", loop=1, train=False)
temp = train_dataset[0]   # debug

traindlr = DataLoader(train_dataset, batch_size=batch_size, 
                      collate_fn=s3dis_collate_fn, shuffle=True, pin_memory=True, 
                      persistent_workers=True, drop_last=True, num_workers=8)
testdlr = DataLoader(test_dataset, batch_size=1,
                      collate_fn=s3dis_collate_fn, pin_memory=True, 
                      persistent_workers=True, num_workers=8)

step_per_epoch = len(traindlr)

# model = DelaSemSeg(dela_args).cuda()
# memory版
model = DelaSemSeg_Mem(dela_args).cuda()
# for finetune
ckpt = torch.load('/mnt/Disk16T/chenhr/DeLA/S3DIS/pretrained/best.pt')
model.load_state_dict(ckpt['model'], strict=False)
lr = lr / 100

optimizer = create_optimizer_v2(model, lr=lr, weight_decay=5e-2)
scheduler = CosineLRScheduler(optimizer, t_initial = epoch * step_per_epoch, lr_min = lr/10000,
                                warmup_t=warmup*step_per_epoch, warmup_lr_init = lr/20)
scaler = GradScaler()
# if wish to continue from a checkpoint
resume = False
if resume:
    start_epoch = util.load_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer, scaler=scaler)["start_epoch"]
else:
    start_epoch = 0

scheduler_step = start_epoch * step_per_epoch

metric = util.Metric(13)
ttls = util.AverageMeter() 
corls = util.AverageMeter() 
best = 0
warmup_fn(model, S3DIS(s3dis_warmup_args, partition="!5", loop=batch_size, warmup=True))
for i in range(start_epoch, epoch):
    model.train()
    ttls.reset()
    corls.reset()
    metric.reset()
    now = time()
    for xyz, feature, indices, pts, y in tqdm(traindlr, desc=f'Epoch {i}/{epoch}'):
        lam = scheduler_step/(epoch*step_per_epoch)
        lam = 3e-3 ** lam * .25
        scheduler.step(scheduler_step)
        scheduler_step += 1
        xyz = xyz.cuda(non_blocking=True)
        feature = feature.cuda(non_blocking=True)
        indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
        pts = pts.tolist()[::-1]
        y = y.cuda(non_blocking=True)
        with autocast():
            # p, closs = model(xyz, feature, indices, pts)
            # memory版
            p, closs, coarse_seg_loss = model(xyz, feature, indices, pts, y, i/epoch)
            loss = F.cross_entropy(p, y, label_smoothing=ls)
        metric.update(p.detach(), y)
        ttls.update(loss.item())
        corls.update(closs.item())
        optimizer.zero_grad(set_to_none=True)
        # scaler.scale(loss + closs*lam).backward()
        # memory版
        scaler.scale(loss + closs*lam + coarse_seg_loss * 0.2).backward()
        scaler.step(optimizer)
        scaler.update()
        
    logger.info(f"epoch {i}:")
    logger.info(f"loss: {round(ttls.avg, 4)} || cls: {round(corls.avg, 4)}")
    metric.print("train:", logger=logger)

    model.eval()
    metric.reset()
    with torch.no_grad():
        for xyz, feature, indices, pts, y in testdlr:
            xyz = xyz.cuda(non_blocking=True)
            feature = feature.cuda(non_blocking=True)
            indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
            y = y.cuda(non_blocking=True)
            with autocast():
                p = model(xyz, feature, indices)
            metric.update(p, y)
    
    metric.print("val:  ", logger=logger)
    logger.info(f"duration: {time() - now}")
    cur = metric.miou
    if best < cur:
        best = cur
        logger.info("new best!")
        util.save_state(f"output/model/{cur_id}/best.pt", model=model)
    
    util.save_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer, scaler=scaler, start_epoch=i+1)