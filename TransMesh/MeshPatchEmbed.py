import torch
import torch.nn as nn
import numpy as np
import open3d
import time
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

class MeshPatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # 1x1x1 convolution in order to apply a matrix to each patch vector
        self.proj = nn.Conv3d(self.in_chans, self.embed_dim, kernel_size=1, stride=1)
    
    def forward(self, x):
        y = self.proj(x)
        return y

class MeshPatchPartition(nn.Module):
    def __init__(self, step, max_dim):
        self.step = step
        self.max_dim = max_dim
    
    def forward(self, x:open3d.geometry.TriangleMesh):
        mesh_bb = x.get_axis_aligned_bounding_box()
        bb_min = mesh_bb.min_bound
        bb_max = mesh_bb.max_bound

        # Define grid for tokens
        xs = np.arange(bb_min[0], bb_max[0], self.step)
        ys = np.arange(bb_min[1], bb_max[1], self.step)
        zs = np.arange(bb_min[2], bb_max[2], self.step)
        tokens = torch.zeros(len(xs),len(ys),len(zs),self.max_dim*12)

        # Tokens is a 3D array where each entry is the patch feature vector
        # We don't ever need to split patches, so now we can treat it like any
        # other volume of data
        for i in range(len(xs)):
            for j in range(len(ys)):
                for k in range(len(zs)):
                    tk = self.make_token(x, np.array([xs[i], ys[j], zs[k]]))
                    tokens[i,j,k] = tk
        # tokens = torch.flatten(tokens, end_dim=2)
        return tokens
    
    def make_tokens(self, trg_obj):
        device = 'cuda'
        # We read the target 3D model using load_obj
        verts, faces, aux = load_obj(trg_obj)

        # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
        # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
        # For this tutorial, normals and textures are ignored.
        faces_idx = faces.verts_idx.to(device)
        verts = verts.to(device)

        # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0). 
        # (scale, center) will be used to bring the predicted mesh to its original center and scale
        # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale

        # We construct a Meshes structure for the target mesh
        trg_mesh = Meshes(verts=[verts], faces=[faces_idx])
        
        tri_pts = torch.reshape(verts[torch.flatten(faces_idx)],(faces_idx.shape[0], 3, 3))
        centroids = tri_pts.mean(axis=1)
        normals = torch.cross(tri_pts[:,1]-tri_pts[:,0], tri_pts[:,2]-tri_pts[:,0])
        areas = 0.5*torch.sqrt(torch.sum(normals.pow(2), 1))
        features = torch.cat((centroids, torch.flatten(tri_pts, start_dim=1), normals, areas))
        buckets = torch.zeros(int(2/self.step), int(2/self.step), int(2/self.step), self.max_dim)

        
    
    def make_token(self, mesh, min_box):
        box = open3d.geometry.AxisAlignedBoundingBox(min_box, min_box+self.step)
        cube = mesh.crop(box)

        vertices = torch.from_numpy(np.asarray(cube.vertices))
        normals = torch.from_numpy(np.asarray(cube.triangle_normals))
        triangles = torch.from_numpy(np.asarray(cube.triangles))
        tri_pts = torch.reshape(vertices[torch.flatten(triangles.long())],(triangles.shape[0], 3, 3))
        centroids = tri_pts.mean(axis=1)

        token = torch.cat((centroids, torch.flatten(tri_pts, start_dim=1)), -1)
        if token.shape[0] <= self.max_dim:
            token = torch.cat((token, torch.zeros(self.max_dim-token.shape[0], token.shape[1])), 0)
        else:
            ab = tri_pts[:,1] - tri_pts[:, 0]
            ac = tri_pts[:,2] - tri_pts[:,0]
            areas = torch.sum(torch.cross(ab, ac).pow(2), 1).reshape(-1, 1)
            sortd, indices = torch.sort(torch.cat((token, areas), 1))
            token = sortd[:self.max_dim, :-1]
        token = torch.flatten(token)
        return token

embed = MeshPatchPartition(0.05, 128)
with open('/playpen/RNNSLAM/window-size-1/031/mesh/test_norm_ave3_031.obj') as f:
    embed.make_tokens(f)

# mesh = open3d.io.read_triangle_mesh('/playpen/RNNSLAM/window-size-1/031/mesh/test_norm_ave3_031.obj')
# tokens = embed.forward(mesh)
# print(tokens.shape)

# pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(centroids.numpy())
# open3d.visualization.draw_geometries([pcd])
