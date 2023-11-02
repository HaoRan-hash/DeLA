import torch
from torch import nn
import torch.nn.functional as F
from scannetv2 import ScanNetV2, scan_test_collate_fn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import utils.util as util
from delasemseg import DelaSemSeg
from delasemseg_mem import DelaSemSeg_Mem
from config import scan_args, dela_args
from torch.cuda.amp import autocast
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np


def gen_color(y):
    """
    y.shape = (n,)
    """
    color_map = [[174, 199, 232], [152, 223, 137], [31, 119, 180],
                 [255, 188, 120], [188, 189, 35], [140, 86, 74],
                 [255, 152, 151], [213, 39, 40], [196, 176, 213],
                 [149, 103, 188], [197, 156, 148], [23, 190, 208], 
                 [247, 182, 210], [219, 219, 141], [255, 127, 14],
                 [158, 218, 230], [43, 160, 45], [112, 128, 144],
                 [227, 119, 194], [82, 83, 163]]
    color_map = np.asarray(color_map, dtype=np.float32)
    res = np.zeros((len(y), 3))
    for i in range(20):
        mask = (y == i)
        res[mask] = color_map[i]
    return res


# torch.set_float32_matmul_precision("high")

parser = ArgumentParser()
parser.add_argument('--save_gt', action='store_true', default=False)
args = parser.parse_args()

# loop x rotation x scaling
loop = 4 * 4 * 3

testdlr = DataLoader(ScanNetV2(scan_args, partition="val", loop=loop, train=False, test=True), batch_size=1,
                      collate_fn=scan_test_collate_fn, pin_memory=True, num_workers=8)

# model = DelaSemSeg(dela_args).cuda()
# memory版
model = DelaSemSeg_Mem(dela_args).cuda()

util.load_state(f"/mnt/Disk16T/chenhr/DeLA/ScanNetV2/output/model/memory_finetune/best.pt", model=model)

model.eval()

cum = 0
cnt = 0
save_dir = Path('/mnt/Disk16T/chenhr/DeLA/ScanNetV2/vis_results/memory')
gt_dir = Path('/mnt/Disk16T/chenhr/DeLA/ScanNetV2/vis_results/gt')
with torch.no_grad():
    for xyz, feature, indices, nn, y, name, full_xyz in tqdm(testdlr):
            xyz = xyz.cuda(non_blocking=True)
            feature = feature.cuda(non_blocking=True)
            indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
            nn = nn.cuda(non_blocking=True).long()
            with autocast(False):
                p = model(xyz, feature, indices)
            cum = cum + p[nn]
            cnt += 1
            if cnt % loop == 0:
                y = y.cuda(non_blocking=True)
                mask = y != 20
                cum, y = cum[mask], y[mask]
                save_file_name = save_dir / (str(name.stem) + '.txt')
                cum = cum.argmax(dim=-1)
                cum = cum.cpu().numpy()
                cum_color = gen_color(cum)
                full_xyz = full_xyz[mask].numpy()
                save_array = np.concatenate((full_xyz, cum_color), axis=1)
                np.savetxt(save_file_name, save_array, fmt='%.4f')   # 分隔符是空格
                if args.save_gt:
                    gt_file_name = gt_dir / (str(name.stem) + '.txt')
                    y = y.cpu().numpy()
                    y_color = gen_color(y)
                    save_array = np.concatenate((full_xyz, y_color), axis=1)
                    np.savetxt(gt_file_name, save_array, fmt='%.4f')
                cnt = cum = 0
