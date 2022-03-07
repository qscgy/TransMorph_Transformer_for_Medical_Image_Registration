import os, glob
import torch, sys
from torch.utils.data import Dataset
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

def norm_center(verts):
    center = verts.mean(0)
    verts = verts - center
    scale = (verts.abs().max(0)[0]).max(0)[0]
    verts = verts / scale
    return verts

class MeshDataset(Dataset):
    def __init__(self, data_path, transforms=None):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]

        device = 'cuda'
        verts_x, faces_x, aux_x = load_obj(path[0])
        verts_y, faces_y, aux_y = load_obj(path[1])

        faces_idx_x = faces_x.verts_idx.to(device)
        verts_x = verts_x.to(device)
        faces_idx_y = faces_y.verts_idx.to(device)
        verts_y = verts_y.to(device)

        verts_x = norm_center(verts_x)
        verts_y = norm_center(verts_y)

        mesh_x = Meshes(verts=[verts_x], faces=[faces_idx_x])
        mesh_y = Meshes(verts=[verts_y], faces=[faces_idx_y])
        return mesh_x, mesh_y

    def __len__(self):
        return len(self.paths)