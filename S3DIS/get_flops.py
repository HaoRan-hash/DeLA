import torch
from delasemseg import DelaSemSeg
from delasemseg_mem import DelaSemSeg_Mem
from s3dis import S3DIS, s3dis_collate_fn
from torch.utils.data import DataLoader
import time
from deepspeed.profiling.flops_profiler import get_model_profile
from config import s3dis_args, dela_args


seed = 4464
torch.manual_seed(seed)

# model = DelaSemSeg(dela_args).cuda()
# memory版
model = DelaSemSeg_Mem(dela_args).cuda()

model.eval()

train_dataset = S3DIS(s3dis_args, partition="!5", loop=1)
traindlr = DataLoader(train_dataset, batch_size=1, 
                      collate_fn=s3dis_collate_fn, shuffle=False, pin_memory=True, 
                      persistent_workers=True, drop_last=True, num_workers=8)

for xyz, feature, indices, pts, y in traindlr:
    if xyz.shape[0] != 24000:
        continue
    else:
        xyz = xyz.cuda(non_blocking=True)
        feature = feature.cuda(non_blocking=True)
        indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
        
        flops, _, params = get_model_profile(model=model, args=[xyz, feature, indices], print_profile=False, detailed=False, warm_up=10,
                                             as_string=False, output_file=None)
        print(f'{params / 1e6}\t{flops / 1e9}')
        
        warmup_iter = 10
        for _ in range(warmup_iter):
            model(xyz, feature, indices)
        
        iter = 200
        st = time.time()
        for _ in range(iter):
            model(xyz, feature, indices)
            torch.cuda.synchronize()
        et = time.time()
        print(f'Inference time(s): {(et-st) / iter}')
        break
