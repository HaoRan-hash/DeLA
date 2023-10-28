import os
import torch
from torch import nn
import torch.nn.functional as F
from s3dis import S3DIS, s3dis_test_collate_fn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import utils.util as util
from delasemseg import DelaSemSeg
from delasemseg_mem import DelaSemSeg_Mem
from config import s3dis_args, dela_args
from torch.cuda.amp import autocast
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--multi_scale', action='store_true', default=False)
args = parser.parse_args()

os.makedirs(f"output/log/six_fold_memory", exist_ok=True)
logfile = "output/log/six_fold_memory/out.log"
logger = util.create_logger(logfile)

# model = DelaSemSeg(dela_args).cuda()
# memory版
model = DelaSemSeg_Mem(dela_args).cuda()

all_metric = util.Metric(13)

for i in range(1, 7):
    loop = 12

    testdlr = DataLoader(S3DIS(s3dis_args, partition=f"{i}", loop=loop, train=False, test=True), batch_size=1,
                        collate_fn=s3dis_test_collate_fn, pin_memory=True, num_workers=8)
    
    util.load_state(f"output/model/memory_0{i}/best.pt", model=model)
    
    model.eval()

    metric = util.Metric(13)
    cum = 0
    cnt = 0
    
    with torch.no_grad():
        for xyz, feature, indices, nn, y in tqdm(testdlr):
                xyz = xyz.cuda(non_blocking=True)
                feature = feature.cuda(non_blocking=True)
                indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
                nn = nn.cuda(non_blocking=True).long()
                with autocast():
                    if args.multi_scale:
                        # voting预测 （做数据增强）
                        multi_scales = [0.9, 1.0, 1.1]
                        for j in range(len(multi_scales)):
                            if j == 0:
                                p = model(xyz * multi_scales[j], feature, indices)
                            else:
                                p += model(xyz * multi_scales[j], feature, indices)
                        p = p / len(multi_scales)
                    else:
                        p = model(xyz, feature, indices)
                    
                cum = cum + p[nn]
                cnt += 1
                if cnt % loop == 0:
                    y = y.cuda(non_blocking=True)
                    metric.update(cum, y)
                    all_metric.update(cum, y)
                    cnt = cum = 0

    metric.print(f"area {i}: ", logger=logger)
all_metric.print("six fold: ", logger=logger)
