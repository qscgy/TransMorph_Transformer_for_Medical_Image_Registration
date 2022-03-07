import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes

class AffineChamferLoss(nn.Module):
    def __init__(self):
        super(AffineChamferLoss, self).__init__()
    
    def forward(self, mesh_fixed, mesh_moving, mat, trans):
        pts_mov = mesh_moving.verts_packed()
        # print(pts_mov.shape)
        # print(trans.shape)
        # print(mat.shape)
        pts_mov_trans = pts_mov@(mat[0]) + torch.t(trans[0])
        return chamfer_distance(mesh_fixed.verts_packed().unsqueeze(0), pts_mov_trans.unsqueeze(0))