import os
import argparse
import torch
import torch.nn.functional as F


import numpy as np
from time import time
from tqdm import tqdm


class GaussFormulaFMM:
    def __init__(self,
                 points: torch.Tensor,
                 widths=None,
                 max_tree_depth=15):
        try:
            import torch_treecode_cpu
            treecode_cpu_loaded = True
        except:
            treecode_cpu_loaded = False
        try:
            import torch_treecode_cuda
            treecode_cuda_loaded = True
        except:
            treecode_cuda_loaded = False
        
        self.is_cuda = points.is_cuda   # undefined behavior if changed after init
        if self.is_cuda:
            assert treecode_cuda_loaded
        else:
            assert treecode_cpu_loaded

        self.treecode_package = torch_treecode_cuda if self.is_cuda else torch_treecode_cpu
        self.device = points.device

        tree_depth = max_tree_depth
        tree_packed = self.treecode_package.build_tree(points.cpu(), tree_depth)   # tree build is on CPU either way

        if self.is_cuda:
            for i in range(len(tree_packed)):
                tree_packed[i] = tree_packed[i].to(self.device)
        node_parent_list, node_children_list, node_is_leaf_list, node_half_w_list, num_points_in_node, node2point_index, node2point_indexstart = tree_packed
        
        if widths is not None:
            self.widths = widths.clone().to(self.device)
        else:
            self.widths = torch.ones(points.shape[0], device=self.device).float() * 0.006    # a somewhat working value, certainly not good enough
        self.node_parent_list = node_parent_list
        self.node_children_list = node_children_list
        self.points = points
        self.node2point_index = node2point_index
        self.node2point_indexstart = node2point_indexstart
        self.num_points_in_node = num_points_in_node
        self.node_is_leaf_list = node_is_leaf_list
        self.node_half_w_list = node_half_w_list
        self.tree_depth = tree_depth

    def forward_A(self, normals, width_scale):
        point_weights = (normals ** 2).sum(-1).sqrt()
        node_normals, node_reppoints, _ = \
            self.treecode_package.scatter_point_attrs_to_nodes(self.node_parent_list,
                                                    self.node_children_list,
                                                    self.points,
                                                    point_weights,
                                                    normals,
                                                    self.node2point_index,
                                                    self.node2point_indexstart,
                                                    self.num_points_in_node,
                                                    self.node_is_leaf_list,
                                                    self.tree_depth)
        out_vals = self.treecode_package.multiply_by_A(
            self.points,
            self.widths * width_scale,
            self.points,
            normals,
            self.node2point_index,
            self.node2point_indexstart,
            self.node_children_list,
            node_normals,
            self.node_is_leaf_list,
            self.node_half_w_list,
            node_reppoints,
            self.num_points_in_node,
        )

        return out_vals
    
    def forward_AT(self, values, width_scale):
        point_weights = (values ** 2).sum(-1).sqrt()
        node_scalars, node_reppoints, _  = \
            self.treecode_package.scatter_point_attrs_to_nodes(self.node_parent_list,
                                                    self.node_children_list,
                                                    self.points,
                                                    point_weights,
                                                    values,
                                                    self.node2point_index,
                                                    self.node2point_indexstart,
                                                    self.num_points_in_node,
                                                    self.node_is_leaf_list,
                                                    self.tree_depth)
        
        out_vecs = self.treecode_package.multiply_by_AT(
            self.points,
            self.widths * width_scale,
            self.points,
            values,
            self.node2point_index,
            self.node2point_indexstart,
            self.node_children_list,
            node_scalars,
            self.node_is_leaf_list,
            self.node_half_w_list,
            node_reppoints,
            self.num_points_in_node,
        )

        return out_vecs
    
    def forward_G(self, normals, width_scale):
        point_weights = (normals ** 2).sum(-1).sqrt()
        node_normals, node_reppoints, _ = \
            self.treecode_package.scatter_point_attrs_to_nodes(self.node_parent_list,
                                                    self.node_children_list,
                                                    self.points,
                                                    point_weights,
                                                    normals,
                                                    self.node2point_index,
                                                    self.node2point_indexstart,
                                                    self.num_points_in_node,
                                                    self.node_is_leaf_list,
                                                    self.tree_depth)
        out_normals = self.treecode_package.multiply_by_G(
            self.points,
            self.widths * width_scale,
            self.points,
            normals,
            self.node2point_index,
            self.node2point_indexstart,
            self.node_children_list,
            node_normals,
            self.node_is_leaf_list,
            self.node_half_w_list,
            node_reppoints,
            self.num_points_in_node,
        )

        return out_normals
    
time_start = time()


parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('--ws', type=float, default=1.0)
parser.add_argument('--wsmax', type=float, default=0.004)
parser.add_argument('--wsmin', type=float, default=0.002)
parser.add_argument('--oiters', type=int, default=30)
parser.add_argument('--out_dir', type=str, default='out_final')
parser.add_argument('--cpu', action='store_true', help='uses gpu by default')
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

points_normals = np.loadtxt(args.input)
points = points_normals[:, :3]

bbox_scale = 1.1
bbox_center = (points.min(0) + points.max(0)) / 2.
bbox_len = (points.max(0) - points.min(0)).max()
points = (points - bbox_center) * (2 / (bbox_len * bbox_scale))

points = torch.from_numpy(points).contiguous().float()
normals = torch.zeros_like(points).contiguous().float()
b = torch.ones(points.shape[0], 1) * 0.5
radii = points[:, 0].clone()    # we support per-point smoothing width, but do not use it in experiments
radii[:] = 1.

if not args.cpu:
    points = points.cuda()
    normals = normals.cuda()
    b = b.cuda()
    radii = radii.cuda()


fmm = GaussFormulaFMM(points, radii)


wsmin = args.wsmin * args.ws
wsmax = args.wsmax * (args.ws ** 2)
if wsmax < wsmin:
    wsmax = wsmin

with torch.no_grad():
    for i in tqdm(range(args.oiters)):
        width_scale = wsmin + ((args.oiters-1-i) / ((args.oiters-1))) * (wsmax - wsmin)
        
        # grad step
        A_mu = fmm.forward_A(normals, width_scale)
        AT_A_mu = fmm.forward_AT(A_mu, width_scale)
        r = fmm.forward_AT(b, width_scale) - AT_A_mu
        A_r = fmm.forward_A(r, width_scale)
        alpha = (r * r).sum() / (A_r * A_r).sum()
        normals = normals + alpha * r

        # WNNC step
        out_normals = fmm.forward_G(normals, width_scale)
        out_normals = F.normalize(out_normals, dim=-1).contiguous()
        normals_len = torch.linalg.norm(normals, dim=-1, keepdim=True)
        out_normals_resized = out_normals * normals_len
        normals = out_normals.clone() * normals_len

    out_normals = fmm.forward_G(normals, width_scale)
    out_normals = F.normalize(out_normals, dim=-1).contiguous()

time_end = time()
print(time_end - time_start)

with torch.no_grad():
    out_points1 = torch.cat([points, out_normals.detach()], -1)
    np.savetxt(os.path.join(args.out_dir, os.path.basename(args.input)[:-4] + f'.xyz'), out_points1.cpu())
