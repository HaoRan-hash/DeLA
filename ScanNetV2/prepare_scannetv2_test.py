import multiprocessing as mp
import plyfile
import torch
import torch.nn.functional as F
from test_config import raw_data_path as src, processed_data_path as dest

print(f"Processed data will be saved in:\n{dest}")

dest.mkdir(exist_ok=True)


def calc_normal(xyz, face):
    x0 = xyz[face[:, 0]]
    x1 = xyz[face[:, 1]].sub_(x0)
    x2 = xyz[face[:, 2]].sub_(x0)
    fn = torch.cross(x1, x2, dim=1)
    face = face.view(-1, 1).expand(-1, 3)
    fn = fn.view(-1, 1, 3).repeat(1, 3, 1).view(-1, 3)
    norm = torch.zeros_like(xyz).scatter_add_(dim=0, index=face, src=fn)
    norm = F.normalize(norm, p=2, dim=1, eps=1e-8)
    return norm

def process_scene(entry):
    scene = entry.name
    f0 = dest / f"{scene}.pt"
    if f0.exists():
        print(f"skipping {scene}")
        return
    f1 = entry / f"{scene}_vh_clean_2.ply"
    f1 = plyfile.PlyData().read(f1)
    face = torch.tensor([e[0].tolist() for e in f1.elements[1]])
    f1 = torch.tensor([e.tolist() for e in f1.elements[0]])
    xyz = f1[:, :3]
    xyz = xyz - xyz.min(dim=0)[0]
    col = f1[:, 3:6].type(torch.uint8)
    norm = calc_normal(xyz, face)
    torch.save((xyz, col, norm), f0)
    print(f"finished {scene}")

entries = [entry for entry in (src / "scans_test").iterdir() if entry.is_dir()]
p = mp.Pool(processes=8)
p.map(process_scene, entries)
p.close()
p.join()