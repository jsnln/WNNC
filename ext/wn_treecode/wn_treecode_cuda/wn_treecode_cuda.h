/*
MIT License

Copyright (c) 2024 Siyou Lin, Zuoqiang Shi, Yebin Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/


#include <vector>

#define ALLOWED_MAX_DEPTH 15
#define SPATIAL_DIM 3
#define NUM_OCT_CHILDREN 8
#define THREADS_PER_BLOCK 1024
#define TREECODE_THRESHOLD 2.0f

typedef long signedindex_t;

// CUDA kernel declarations
template<typename scalar_t>
__global__ void scatter_point_attrs_to_nodes_leaf_cuda_kernel(
    const signedindex_t* ptr_node_parent_list,
    const scalar_t* ptr_points,
    const scalar_t* ptr_point_weights,
    const scalar_t* ptr_point_attrs,
    const signedindex_t* ptr_node2point_index,
    const signedindex_t* ptr_node2point_indexstart,
    const signedindex_t* ptr_num_points_in_node,
    const bool* ptr_node_is_leaf_list,
    bool* ptr_scattered_mask,

    scalar_t* ptr_out_node_attrs,
    scalar_t* ptr_out_node_reppoints,
    scalar_t* ptr_out_node_weights,

    signedindex_t attr_dim,
    signedindex_t num_nodes
);


template<typename scalar_t>
__global__ void scatter_point_attrs_to_nodes_nonleaf_cuda_kernel(
    const signedindex_t* ptr_node_parent_list,
    const signedindex_t* ptr_node_children_list,
    const scalar_t* ptr_points,
    const scalar_t* ptr_point_weights,
    const scalar_t* ptr_point_attrs,
    const signedindex_t* ptr_node2point_index,
    const signedindex_t* ptr_node2point_indexstart,
    const signedindex_t* ptr_num_points_in_node,
    const bool* ptr_node_is_leaf_list,
    bool* ptr_scattered_mask,
    const bool* ptr_next_to_scatter_mask,

    scalar_t* ptr_out_node_attrs,
    scalar_t* ptr_out_node_reppoints,
    scalar_t* ptr_out_node_weights,

    signedindex_t attr_dim,
    signedindex_t num_nodes
);

template<typename scalar_t>
__global__ void find_next_to_scatter_cuda_kernel(
    const signedindex_t* ptr_node_children_list,
    const bool* ptr_node_is_leaf_list,
    bool* ptr_scattered_mask,
    bool* ptr_next_to_scatter_mask,
    const signedindex_t* node2point_index,
    signedindex_t num_nodes
);

template<typename scalar_t>
__global__ void multiply_by_A_cuda_kernel(
    const scalar_t* query_points,  // [N', 3]
    const scalar_t* query_width,   // [N',]
    const scalar_t* points,        // [N, 3]
    const scalar_t* point_attrs,   // [N, C]
    const signedindex_t* node2point_index,
    const signedindex_t* node2point_indexstart,
    const signedindex_t* node_children_list,
    const scalar_t* node_attrs,
    const bool* node_is_leaf_list,
    const scalar_t* node_half_w_list,
    const scalar_t* node_reppoints,
    const signedindex_t* num_points_in_node,
    scalar_t* out_attrs,           // [N,]
    signedindex_t num_queries,
    bool continuous=false
);


template<typename scalar_t>
__global__ void multiply_by_AT_cuda_kernel(
    const scalar_t* query_points,  // [N', 3]
    const scalar_t* query_width,   // [N',]
    const scalar_t* points,        // [N, 3]
    const scalar_t* point_attrs,   // [N, C]
    const signedindex_t* node2point_index,
    const signedindex_t* node2point_indexstart,
    const signedindex_t* node_children_list,
    const scalar_t* node_attrs,
    const bool* node_is_leaf_list,
    const scalar_t* node_half_w_list,
    const scalar_t* node_reppoints,
    const signedindex_t* num_points_in_node,
    scalar_t* out_attrs,           // [N, 3]
    signedindex_t num_queries
);

template<typename scalar_t>
__global__ void multiply_by_G_cuda_kernel(
    const scalar_t* query_points,  // [N', 3]
    const scalar_t* query_width,   // [N',]
    const scalar_t* points,        // [N, 3]
    const scalar_t* point_attrs,   // [N, C]
    const signedindex_t* node2point_index,
    const signedindex_t* node2point_indexstart,
    const signedindex_t* node_children_list,
    const scalar_t* node_attrs,
    const bool* node_is_leaf_list,
    const scalar_t* node_half_w_list,
    const scalar_t* node_reppoints,
    const signedindex_t* num_points_in_node,
    scalar_t* out_attrs,           // [N, 3]
    signedindex_t num_queries
);