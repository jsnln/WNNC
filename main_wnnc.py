import os
import psutil
import argparse
import torch
import torch.nn.functional as F
import math

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
parser.add_argument('-s', '--save_intermediate', nargs='*', type=int)
parser.add_argument('--ws', type=float, default=1.0)
# [0.04, 0.16]
# [0.03, 0.12]
# [0.02, 0.08]
# [0.01, 0.04] real, 100 iters
# [0.05, 0.2] sparse and sketch
parser.add_argument('--wsmax', type=float, default=0.016)
parser.add_argument('--wsmin', type=float, default=0.002)
parser.add_argument('--oiters', type=int, default=40)
parser.add_argument('--out_dir', type=str, default='out_final')
parser.add_argument('--cpu', action='store_true', help='uses gpu by default')
parser.add_argument('--tqdm', action='store_true', help='use tqdm bar')
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)


if os.path.splitext(args.input)[-1] == '.xyz':
    points_normals = np.loadtxt(args.input)
    points_unnormalized = points_normals[:, :3]
if os.path.splitext(args.input)[-1] in ['.ply', '.obj']:
    import trimesh
    pcd = trimesh.load(args.input)
    points_unnormalized = np.array(pcd.vertices)
if os.path.splitext(args.input)[-1] == '.npy':
    pcd = np.load(args.input)
    points_unnormalized = pcd[:, :3]

time_preprocess_start = time()

bbox_scale = 1.1
bbox_center = (points_unnormalized.min(0) + points_unnormalized.max(0)) / 2.
bbox_len = (points_unnormalized.max(0) - points_unnormalized.min(0)).max()
points_normalized = (points_unnormalized - bbox_center) * (2 / (bbox_len * bbox_scale))

points_normalized = torch.from_numpy(points_normalized).contiguous().float()
normals = torch.zeros_like(points_normalized).contiguous().float()
b = torch.ones(points_normalized.shape[0], 1) * 0.5
radii = points_normalized[:, 0].clone()    # we support per-point smoothing width, but do not use it in experiments
radii[:] = 1.

if not args.cpu:
    points_normalized = points_normalized.cuda()
    normals = normals.cuda()
    b = b.cuda()
    radii = radii.cuda()


fmm = GaussFormulaFMM(points_normalized, radii)


wsmin = args.wsmin * args.ws
# wsmax = args.wsmax * (args.ws ** 2)
wsmax = args.wsmax * args.ws
if wsmax < wsmin:
    wsmax = wsmin

if fmm.is_cuda:
    torch.cuda.synchronize(device=None)

time_iter_start = time()
with torch.no_grad():
    bar = tqdm(range(args.oiters)) if args.tqdm else range(args.oiters)

    for i in bar:
        width_scale = wsmin + ((args.oiters-1-i) / ((args.oiters-1))) * (wsmax - wsmin)
        # width_scale = args.wsmin + 0.5 * (args.wsmax - args.wsmin) * (1 + math.cos(i/(args.oiters-1) * math.pi))
        
        # grad step
        A_mu = fmm.forward_A(normals, width_scale)
        AT_A_mu = fmm.forward_AT(A_mu, width_scale)
        r = fmm.forward_AT(b, width_scale) - AT_A_mu
        A_r = fmm.forward_A(r, width_scale)
        alpha = (r * r).sum() / (A_r * A_r).sum()
        normals = normals + alpha * r

        # WNNC step
        out_normals = fmm.forward_G(normals, width_scale)

        # rescale
        out_normals = F.normalize(out_normals, dim=-1).contiguous()
        normals_len = torch.linalg.norm(normals, dim=-1, keepdim=True)
        normals = out_normals.clone() * normals_len

        if fmm.is_cuda:
            torch.cuda.synchronize(device=None)
        
        # time_now = time()
        # if args.save_intermediate is not None and i in args.save_intermediate:
        #     print(f'[LOG] iter: {i}, time: {time_now - time_iter_start:.5f}')

        
        # process = psutil.Process(os.getpid())
        # mem_info = process.memory_info()    # bytes
        # print('mem:', mem_info.rss / 1024/1024)     # megabytes

    # out_normals = fmm.forward_G(normals, width_scale)
    # out_normals = F.normalize(out_normals, dim=-1).contiguous()

time_iter_end = time()
print(f'[LOG] time_preproc: {time_iter_start - time_preprocess_start}')
print(f'[LOG] time_main: {time_iter_end - time_iter_start}')

with torch.no_grad():
    out_points1 = np.concatenate([points_unnormalized, out_normals.detach().cpu().numpy()], -1)
    np.savetxt(os.path.join(args.out_dir, os.path.basename(args.input)[:-4] + f'.xyz'), out_points1)

process = psutil.Process(os.getpid())
mem_info = process.memory_info()    # bytes
mem = mem_info.rss
if fmm.is_cuda:
    gpu_mem = torch.cuda.mem_get_info(0)[1]-torch.cuda.mem_get_info(0)[0]
    mem += gpu_mem
print('[LOG] mem:', mem / 1024/1024)     # megabytes