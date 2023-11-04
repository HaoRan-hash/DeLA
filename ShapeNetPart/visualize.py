import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from shapenetpart import PartTest, PartNormalDataset
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import utils.util as util
from delapartseg import DelaPartSeg
from delapartseg_mem import DelaPartSeg_Mem
from config import dela_args, object_to_part
import numpy as np
from putil import cls2parts, part_seg_refinement
from tqdm import tqdm
from argparse import ArgumentParser


def gen_color(y):
    """
    y.shape = (n,)
    """
    color_map = [[0, 255, 0], [0, 0, 255],
                 [107, 142, 35], [255, 0, 255],
                 [255, 165, 0], [153, 50, 204]]
    color_map = np.asarray(color_map, dtype=np.float32)
    res = np.zeros((len(y), 3))
    for i in range(6):
        mask = (y == i)
        res[mask] = color_map[i]
    return res


# torch.set_float32_matmul_precision("high")

parser = ArgumentParser()
parser.add_argument('--save_gt', action='store_true', default=False)
args = parser.parse_args()

train_dataset = PartNormalDataset()   # for object_to_part_onehot
test_dataset = PartTest()
testdlr = DataLoader(test_dataset, batch_size=1,
                      pin_memory=True, num_workers=6)
idx_to_class = test_dataset.idx_to_class

# model = DelaPartSeg(dela_args).cuda()
# memory版
model = DelaPartSeg_Mem(dela_args).cuda()
util.load_state("/mnt/Disk16T/chenhr/DeLA/ShapeNetPart/output/model/memory_finetune_03/best.pt", model=model)
model.eval()

save_dir = Path('/mnt/Disk16T/chenhr/DeLA/ShapeNetPart/vis_results/memory')
gt_dir = Path('/mnt/Disk16T/chenhr/DeLA/ShapeNetPart/vis_results/gt')
with torch.no_grad():
    for j, (xyz, norm, shape, y) in enumerate(tqdm(testdlr)):
        B, N, _ = xyz.shape
        xyz = xyz.cuda(non_blocking=True)
        shape = shape.cuda(non_blocking=True)
        norm = norm.cuda(non_blocking=True)
        mask = train_dataset.object_to_part_onehot[shape]   # for memory
        logits = 0
        for i in range(10):
            ixyz = xyz
            inorm = norm
            scale = torch.rand((3,), device=xyz.device) * 0.4 + 0.8
            ixyz = ixyz * scale
            inorm = inorm * (scale[[1, 2, 0]] * scale[[2, 0, 1]])
            inorm = F.normalize(inorm, p=2, dim=-1, eps=1e-8)
            with autocast():
                # logits  = logits + model(ixyz, inorm, shape)
                # memory版
                logits = logits + model(ixyz, inorm, shape, mask)
        
        p = logits.max(dim=1)[1].view(B, N)
        part_seg_refinement(p, xyz, shape, cls2parts, 10)
        
        shape_name = idx_to_class[shape.item()]
        cur_dir = save_dir / shape_name
        if not cur_dir.exists():
            cur_dir.mkdir(mode=0o755)
        save_file_name = cur_dir / (shape_name + f'_{j}.txt')
        p = p.squeeze().cpu().numpy()
        p = p - object_to_part[shape.item()][0]   # 做差
        xyz = xyz.squeeze().cpu().numpy()
        p_color = gen_color(p)
        save_array = np.concatenate((xyz, p_color), axis=1)
        np.savetxt(save_file_name, save_array, fmt='%.4f')   # 分隔符是空格
        if args.save_gt:
            cur_dir = gt_dir / shape_name
            if not cur_dir.exists():
                cur_dir.mkdir(mode=0o755)
            gt_file_name = cur_dir / (shape_name + f'_{j}.txt')
            y = y.squeeze().numpy()
            y = y - object_to_part[shape.item()][0]   # 做差
            y_color = gen_color(y)
            save_array = np.concatenate((xyz, y_color), axis=1)
            np.savetxt(gt_file_name, save_array, fmt='%.4f')   # 分隔符是空格
