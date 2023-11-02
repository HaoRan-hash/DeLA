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
import numpy as np


def gen_color(y):
    """
    y.shape = (n,)
    """
    color_map = [[0, 255, 0], [0, 0, 255], [136, 206, 250],
                 [255, 255, 0], [255, 0, 255], [219, 90, 107],
                 [107, 142, 35], [255, 165, 0], [153, 50, 204],
                 [139, 26, 26], [0, 100, 0], [156, 156, 156], [0, 0, 0]]
    color_map = np.asarray(color_map, dtype=np.float32)
    res = np.zeros((len(y), 3))
    for i in range(13):
        mask = (y == i)
        res[mask] = color_map[i]
    return res


# torch.set_float32_matmul_precision("high")

parser = ArgumentParser()
parser.add_argument('--multi_scale', action='store_true', default=False)
parser.add_argument('--save_gt', action='store_true', default=False)
args = parser.parse_args()

loop = 12

testdlr = DataLoader(S3DIS(s3dis_args, partition="5", loop=loop, train=False, test=True), batch_size=1,
                      collate_fn=s3dis_test_collate_fn, pin_memory=True, num_workers=8)

# model = DelaSemSeg(dela_args).cuda()
# memory版
model = DelaSemSeg_Mem(dela_args).cuda()

util.load_state(f"/mnt/Disk16T/chenhr/DeLA/S3DIS/output/model/memory_05/best.pt", model=model)

model.eval()

cum = 0
cnt = 0
save_dir = Path('/mnt/Disk16T/chenhr/DeLA/S3DIS/vis_results/memory')
gt_dir = Path('/mnt/Disk16T/chenhr/DeLA/S3DIS/vis_results/gt')
with torch.no_grad():
    for xyz, feature, indices, nn, y, name, full_xyz in tqdm(testdlr):
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
                save_file_name = save_dir / (str(name.stem) + '.txt')
                cum = cum.argmax(dim=-1)
                cum = cum.cpu().numpy()
                cum_color = gen_color(cum)
                full_xyz = full_xyz.numpy()
                save_array = np.concatenate((full_xyz, cum_color), axis=1)
                np.savetxt(save_file_name, save_array, fmt='%.4f')   # 分隔符是空格
                if args.save_gt:
                    gt_file_name = gt_dir / (str(name.stem) + '.txt')
                    y = y.numpy()
                    y_color = gen_color(y)
                    save_array = np.concatenate((full_xyz, y_color), axis=1)
                    np.savetxt(gt_file_name, save_array, fmt='%.4f')
                cnt = cum = 0

