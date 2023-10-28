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

# torch.set_float32_matmul_precision("high")

parser = ArgumentParser()
parser.add_argument('--cur_id', required=True)
parser.add_argument('--multi_scale', action='store_true', default=False)
args = parser.parse_args()

cur_id = args.cur_id
# logfile = f"output/log/{cur_id}/out.log"
logfile = "pretrained/out.log"
logger = util.create_logger(logfile)

loop = 12

testdlr = DataLoader(S3DIS(s3dis_args, partition="5", loop=loop, train=False, test=True), batch_size=1,
                      collate_fn=s3dis_test_collate_fn, pin_memory=True, num_workers=8)

# model = DelaSemSeg(dela_args).cuda()
# memory版
model = DelaSemSeg_Mem(dela_args).cuda()

util.load_state(f"output/model/{cur_id}/best.pt", model=model)
# util.load_state(f"pretrained/best.pt", model=model)

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
                    for i in range(len(multi_scales)):
                        if i == 0:
                            p = model(xyz * multi_scales[i], feature, indices)
                        else:
                            p += model(xyz * multi_scales[i], feature, indices)
                    p = p / len(multi_scales)
                else:
                    p = model(xyz, feature, indices)
                
            cum = cum + p[nn]
            cnt += 1
            if cnt % loop == 0:
                y = y.cuda(non_blocking=True)
                metric.update(cum, y)
                cnt = cum = 0

metric.print("test: ", logger=logger)
