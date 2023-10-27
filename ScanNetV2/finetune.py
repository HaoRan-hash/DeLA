import random, os
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from scannetv2 import ScanNetV2, scan_collate_fn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.scheduler.cosine_lr import CosineLRScheduler
from utils.timm.optim import create_optimizer_v2
import utils.util as util
from delasemseg import DelaSemSeg
from delasemseg_mem import DelaSemSeg_Mem
from time import time, sleep
from config import scan_args, scan_warmup_args, dela_args, batch_size, learning_rate as lr, epoch, warmup, label_smoothing as ls
from tqdm import tqdm
from argparse import ArgumentParser
from utils.lovasz_loss import lovasz_softmax

# torch.set_float32_matmul_precision("high")

def warmup_fn(model, dataset):
    model.train()
    traindlr = DataLoader(dataset, batch_size=len(dataset), collate_fn=scan_collate_fn, pin_memory=True, num_workers=6)
    for xyz, feature, indices, pts, y in traindlr:
        xyz = xyz.cuda(non_blocking=True)
        feature = feature.cuda(non_blocking=True)
        indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
        pts = pts.tolist()[::-1]
        y = y.cuda(non_blocking=True)
        with autocast():
            # p, closs = model(xyz, feature, indices, pts)
            # loss = F.cross_entropy(p, y, label_smoothing=ls, ignore_index=20) + closs
            # memory版
            p, closs, coarse_seg_loss = model(xyz, feature, indices, pts, y, 0)
            lovasz_loss = lovasz_softmax(p.softmax(dim=-1), y, ignore=20)
            loss = F.cross_entropy(p, y, label_smoothing=ls, ignore_index=20) + closs + coarse_seg_loss * 0.1 + lovasz_loss * 3
        loss.backward()

parser = ArgumentParser()
parser.add_argument('--cur_id', required=True)
args = parser.parse_args()

cur_id = args.cur_id
os.makedirs(f"output/log/{cur_id}", exist_ok=True)
os.makedirs(f"output/model/{cur_id}", exist_ok=True)
logfile = f"output/log/{cur_id}/out.log"
logger = util.create_logger(logfile)

logger.info(r"base")

train_dataset = ScanNetV2(scan_args, partition="train", loop=6)
test_dataset = ScanNetV2(scan_args, partition="val", loop=1, train=False)
temp = train_dataset[0]   # debug

traindlr = DataLoader(train_dataset, batch_size=batch_size, 
                      collate_fn=scan_collate_fn, shuffle=True, pin_memory=True, 
                      persistent_workers=True, drop_last=True, num_workers=8)
testdlr = DataLoader(test_dataset, batch_size=1,
                      collate_fn=scan_collate_fn, pin_memory=True, 
                      persistent_workers=True, num_workers=8)

step_per_epoch = len(traindlr)

# model = DelaSemSeg(dela_args).cuda()
# memory版
model = DelaSemSeg_Mem(dela_args).cuda()
# for finetune
ckpt = torch.load('/mnt/Disk16T/chenhr/DeLA/ScanNetV2/pretrained/best.pt')
model.load_state_dict(ckpt['model'], strict=False)
lr = lr / 100
epoch = 30

optimizer = create_optimizer_v2(model, lr=lr, weight_decay=5e-2)
scheduler = CosineLRScheduler(optimizer, t_initial = epoch * step_per_epoch, lr_min = lr/10000)
scaler = GradScaler()
# if wish to continue from a checkpoint
resume = False
if resume:
    start_epoch = util.load_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer, scaler=scaler)["start_epoch"]
else:
    start_epoch = 0

scheduler_step = start_epoch * step_per_epoch

metric = util.Metric(20)
ttls = util.AverageMeter() 
corls = util.AverageMeter() 
best = 0
warmup_fn(model, ScanNetV2(scan_warmup_args, partition="train", loop=batch_size, warmup=True))
for i in range(start_epoch, epoch):
    model.train()
    ttls.reset()
    metric.reset()
    corls.reset()
    now = time()
    for xyz, feature, indices, pts, y in tqdm(traindlr, desc=f'Epoch {i}/{epoch}'):
        lam = scheduler_step/(epoch*step_per_epoch)
        lam = 3e-3 ** lam * 0.2
        scheduler.step(scheduler_step)
        scheduler_step += 1
        xyz = xyz.cuda(non_blocking=True)
        feature = feature.cuda(non_blocking=True)
        indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
        pts = pts.tolist()[::-1]
        y = y.cuda(non_blocking=True)
        mask = y != 20
        with autocast():
            # p, closs = model(xyz, feature, indices, pts)
            # memory版
            p, closs, coarse_seg_loss = model(xyz, feature, indices, pts, y, i/epoch)
            lovasz_loss = lovasz_softmax(p.softmax(dim=-1), y, ignore=20)
            loss = F.cross_entropy(p, y, label_smoothing=ls, ignore_index=20)
        metric.update(p.detach()[mask], y[mask])
        ttls.update(loss.item())
        corls.update(closs.item())
        optimizer.zero_grad(set_to_none=True)
        # scaler.scale(loss + lam*closs).backward()
        # memory版
        scaler.scale(loss + closs*lam + coarse_seg_loss * 0.1 + lovasz_loss * 3).backward()
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
            mask = y != 20
            with autocast():
                p = model(xyz, feature, indices)
            metric.update(p[mask], y[mask])
    
    metric.print("val:  ", logger=logger)
    logger.info(f"duration: {time() - now}")
    cur = metric.miou
    if best < cur:
        best = cur
        logger.info("new best!")
        util.save_state(f"output/model/{cur_id}/best.pt", model=model)
    
    util.save_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer, scaler=scaler, start_epoch=i+1)
