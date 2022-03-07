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

def three_d_list(l, dim):
    if l==0:
        return [[] for i in range(dim[-1])]
    else:
        return [three_d_list(l-1, dim) for i in range(dim[-l-1])]

class MeshPatchEmbed(nn.Module):
    def __init__(self, step, max_dim, in_chans, embed_dim):
        super(MeshPatchEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.partition = MeshPatchPartition(step, max_dim)

        # 1x1x1 convolution in order to apply a matrix to each patch vector
        self.proj = nn.Conv3d(self.in_chans, self.embed_dim, kernel_size=1, stride=1)
    
    def forward(self, x):
        y = self.partition._forward_open3d(x).cuda()
        y = torch.permute(y, (3, 0, 1, 2))
        y = torch.unsqueeze(y, 0)
        y = self.proj(y)
        return y

class MeshPatchPartition(nn.Module):
    def __init__(self, step, max_dim):
        super(MeshPatchPartition, self).__init__()
        self.step = step
        self.max_dim = max_dim
    
    def forward(self, x):
        tokens = self._forward_open3d(x)
        return tokens

    # Use open3d to do the tokenization since it has an actual crop method
    # x is a pytorch3d mesh
    def _forward_open3d(self, x):
        verts = open3d.utility.Vector3dVector(x.verts_packed().detach().cpu().numpy())
        faces_idx = open3d.utility.Vector3iVector(x.faces_packed().detach().cpu().numpy())
        x = open3d.geometry.TriangleMesh(verts, faces_idx)
        mesh_bb = x.get_axis_aligned_bounding_box()
        bb_min = mesh_bb.min_bound
        bb_max = mesh_bb.max_bound

        # Define grid for tokens
        xs = np.arange(-1, 1, self.step)
        ys = np.arange(-1, 1, self.step)
        zs = np.arange(-1, 1, self.step)
        tokens = torch.zeros(len(xs),len(ys),len(zs),self.max_dim*9)

        # trg_mesh = load_mesh()

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

    def make_tokens(self, mesh):
        verts = mesh.verts_packed()
        faces_idx = mesh.faces_packed()
        tri_pts = torch.reshape(verts[torch.flatten(faces_idx)],(faces_idx.shape[0], 3, 3))
        centroids = tri_pts.mean(axis=1)
        # normals = torch.cross(tri_pts[:,1]-tri_pts[:,0], tri_pts[:,2]-tri_pts[:,0])
        # areas = 0.5*torch.sqrt(torch.sum(normals.pow(2), 1))
        areas = mesh.faces_areas_packed().reshape(-1,1)

        # features = torch.cat((centroids, torch.flatten(tri_pts, start_dim=1), normals, areas))
        features = torch.cat((torch.flatten(tri_pts, start_dim=1), areas), dim=-1)
        print(features.shape)
        buckets = torch.zeros((int(2/self.step), int(2/self.step), int(2/self.step), self.max_dim, features.shape[-1]))
        smallest = torch.zeros((int(2/self.step), int(2/self.step), int(2/self.step)), dtype=torch.int32)
        head = torch.zeros_like(smallest, dtype=torch.int32)
        for i in range(centroids.shape[0]):
            ct = centroids[i]
            bi = (ct/self.step).long()
            head_idx = head[bi[0], bi[1], bi[2]]
            if head_idx < self.max_dim:
                buckets[bi[0], bi[1], bi[2], head_idx] = features[i]
                if features[i,-1] < buckets[bi[0], bi[1], bi[2], smallest[bi[0], bi[1], bi[2]], -1]:
                    smallest[bi[0], bi[1], bi[2]] = head_idx
                head[bi[0], bi[1], bi[2]] += 1
            elif features[i,-1] > buckets[bi[0], bi[1], bi[2], smallest[bi[0], bi[1], bi[2]], -1]:
                buckets[bi[0], bi[1], bi[2], smallest[bi[0], bi[1], bi[2]]] = features[i]
                for j in range(self.max_dim):
                    if buckets[bi[0], bi[1], bi[2], j, -1] < buckets[bi[0], bi[1], bi[2], smallest[bi[0], bi[1], bi[2]], -1]:
                        smallest[bi[0], bi[1], bi[2]] = j
            

    def make_token(self, mesh, min_box):
        box = open3d.geometry.AxisAlignedBoundingBox(min_box, min_box+self.step)
        cube = mesh.crop(box)

        vertices = torch.from_numpy(np.asarray(cube.vertices))
        normals = torch.from_numpy(np.asarray(cube.triangle_normals))
        triangles = torch.from_numpy(np.asarray(cube.triangles))
        tri_pts = torch.reshape(vertices[torch.flatten(triangles.long())],(triangles.shape[0], 3, 3))
        centroids = tri_pts.mean(axis=1)

        # token = torch.cat((centroids, torch.flatten(tri_pts, start_dim=1)), -1)
        token = torch.flatten(tri_pts, start_dim=1)
        if token.shape[0] <= self.max_dim:
            token = torch.cat((token, torch.zeros(self.max_dim-token.shape[0], token.shape[1])), 0)
        else:
            ab = tri_pts[:,1] - tri_pts[:, 0]
            ac = tri_pts[:,2] - tri_pts[:,0]
            areas = torch.sum(torch.cross(ab, ac).pow(2), 1).reshape(-1, 1)
            _, indices = torch.sort(areas)
            token = token[indices[:self.max_dim].long()]
        token = torch.flatten(token)
        return token

def load_mesh(trg_obj):
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
    return trg_mesh

if __name__=='__main__':
    # lis = three_d_list(2, (2,3,4))
    # print(lis[1][1][1])

    embed = MeshPatchEmbed(1/8, 128, 128*9, 512)
    embed.cuda()
    mesh = load_mesh('/playpen/meshes-better/sim_082.obj')
    # tokens = embed.forward(mesh)
    # print(tokens.shape)

    # mesh = open3d.io.read_triangle_mesh('/playpen/RNNSLAM/window-size-1/031/mesh/test_norm_ave3_031.obj')
    tokens = embed(mesh)
    print(tokens.shape)

    # mesh = open3d.io.read_triangle_mesh('/playpen/RNNSLAM/window-size-1/031/mesh/test_norm_ave3_031.obj')
    # tokens = embed.forward(mesh)
    # print(tokens.shape)

    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(centroids.numpy())
    # open3d.visualization.draw_geometries([pcd])
