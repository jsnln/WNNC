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

//////////////////// tree utils ////////////////////
template<typename scalar_t>
struct OctNode;

template<typename scalar_t>
OctNode<scalar_t>* build_tree_cpu_recursive(
    const scalar_t* point_coords,
    const std::vector<signedindex_t>& point_indices,
    OctNode<scalar_t>* const parent,
    scalar_t c_x, scalar_t c_y, scalar_t c_z, scalar_t half_w,
    signedindex_t cur_depth,
    signedindex_t & cur_node_index,
    const signedindex_t & max_depth,
    const signedindex_t & max_points_per_node
);

template<typename scalar_t>
void serialize_tree_recursive(
    OctNode<scalar_t>* cur_node,
    signedindex_t* ptr_node_parent_list,
    signedindex_t* ptr_node_children_list,
    bool* ptr_node_is_leaf_list,
    scalar_t* ptr_node_half_w_list,
    signedindex_t* ptr_num_points_in_node,
    signedindex_t* ptr_node2point_indexstart,
    std::vector<signedindex_t>& stdvec_node2point_index
);


template<typename scalar_t>
void compute_tree_attributes(
    const OctNode<scalar_t> * cur_node,
    signedindex_t & num_nodes,
    signedindex_t & num_leaves,
    signedindex_t & depth
);

template<typename scalar_t>
void free_tree_recursive(OctNode<scalar_t>* cur_node);


//////////////////// treecode op wrappers ////////////////////
template<typename scalar_t>
void scatter_point_attrs_to_nodes_leaf_cpu_kernel_launcher(
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
void scatter_point_attrs_to_nodes_nonleaf_cpu_kernel_launcher(
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
void find_next_to_scatter_cpu_kernel_launcher(
    const signedindex_t* ptr_node_children_list,
    const bool* ptr_node_is_leaf_list,
    bool* ptr_scattered_mask,
    bool* ptr_next_to_scatter_mask,
    const signedindex_t* node2point_index,
    signedindex_t num_nodes
);


template<typename scalar_t>
void multiply_by_A_cpu_kernel_launcher(
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
    bool continuous_kernel=false
);


template<typename scalar_t>
void multiply_by_AT_cpu_kernel_launcher(
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
void multiply_by_G_cpu_kernel_launcher(
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

