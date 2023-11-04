from types import SimpleNamespace
from pathlib import Path
from torch import nn
import torch

# ShapeNetPart dataset path
data_path = Path("/mnt/Disk16T/chenhr/threed_data/data/shapenetcore_partanno_segmentation_benchmark_v0_normal")

presample_path = data_path.parent / "shapenet_part_presample.pt"

epoch = 250
warmup = 20
batch_size = 32
learning_rate = 2e-3
label_smoothing = 0.2

dela_args = SimpleNamespace()
dela_args.depths = [4, 4, 4, 4]
dela_args.ns = [2048, 512, 192, 64]
dela_args.ks = [20, 20, 20, 20]
dela_args.dims = [96, 192, 320, 512]
dela_args.nbr_dims = [48,48]  
dela_args.head_dim = 320
dela_args.num_classes = 50
dela_args.shape_classes = 16
drop_path = 0.15
drop_rates = torch.linspace(0., drop_path, sum(dela_args.depths)).split(dela_args.depths)
dela_args.drop_paths = [dpr.tolist() for dpr in drop_rates]
dela_args.head_drops = torch.linspace(0., 0.15, len(dela_args.depths)).tolist()
dela_args.bn_momentum = 0.1
dela_args.act = nn.GELU
dela_args.mlp_ratio = 2
dela_args.cor_std = [0.75, 1.5, 2.5, 4.7]

dela_args.memory_depth = [3]
dela_args.memory_length = 32
object_to_part = {0: [0, 1, 2, 3], 1: [4, 5], 2: [6, 7], 3: [8, 9, 10, 11], 4: [12, 13, 14, 15], 5: [16, 17, 18], 
                  6: [19, 20, 21], 7: [22, 23], 8: [24, 25, 26, 27], 9: [28, 29], 10: [30, 31, 32, 33, 34, 35],
                  11: [36, 37], 12: [38, 39, 40], 13: [41, 42, 43], 14: [44, 45, 46], 15: [47, 48, 49]}
