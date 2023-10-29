import torch
from torch import nn
import torch.nn.functional as F
from scannetv2 import ScanNetV2_Test, scan_test_collate_fn
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

# torch.set_float32_matmul_precision("high")

# loop x rotation x scaling
loop = 4 * 4 * 3
test_dataset = ScanNetV2_Test(scan_args, data_path='/mnt/Disk16T/chenhr/threed_data/data/scannetv2_test', loop=loop)

testdlr = DataLoader(test_dataset, batch_size=1,
                      collate_fn=scan_test_collate_fn, pin_memory=True, num_workers=8)

model = DelaSemSeg(dela_args).cuda()
# memory版
# model = DelaSemSeg_Mem(dela_args).cuda()

util.load_state('/mnt/Disk16T/chenhr/DeLA/ScanNetV2/pretrained/best.pt', model=model)

model.eval()

cum = 0
cnt = 0
save_dir = Path('/mnt/Disk16T/chenhr/DeLA/ScanNetV2/save_zips/dela')
label_mapping={0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 14, 13: 16, 14: 24, 15: 28, 16: 33, 17: 34, 18: 36, 19: 39}
with torch.no_grad():
    for xyz, feature, indices, nn, name in tqdm(testdlr):
            xyz = xyz.cuda(non_blocking=True)
            feature = feature.cuda(non_blocking=True)
            indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
            nn = nn.cuda(non_blocking=True).long()
            with autocast(False):
                p = model(xyz, feature, indices)
            cum = cum + p[nn]
            cnt += 1
            if cnt % loop == 0:
                save_file_name = save_dir / (str(name.stem) + '.txt')
                cum = cum.argmax(dim=-1)
                cum = cum.cpu().numpy()
                cum = np.vectorize(label_mapping.get)(cum)
                np.savetxt(save_file_name, cum, fmt="%d")
                cnt = cum = 0
