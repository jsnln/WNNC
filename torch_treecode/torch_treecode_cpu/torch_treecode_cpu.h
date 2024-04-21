#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_FOR_CUDA(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_FOR_CPU(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

#define ALLOWED_MAX_DEPTH 15
#define SPATIAL_DIM 3
#define NUM_OCT_CHILDREN 8
#define THREADS_PER_BLOCK 1024
#define TREECODE_THRESHOLD 2.0f

typedef long signedindex_t;


// CUDA forward declarations


std::vector<torch::Tensor> scatter_point_attrs_to_nodes_cpu(
    // torch::Tensor node_parent_list,
    torch::Tensor node_parent_list,
    torch::Tensor node_children_list,
    torch::Tensor points,
    torch::Tensor point_weights,
    torch::Tensor point_attrs,
    torch::Tensor node2point_index,
    torch::Tensor node2point_indexstart,
    torch::Tensor num_points_in_node,
    torch::Tensor node_is_leaf_list,
    signedindex_t tree_depth
);

torch::Tensor multiply_by_A_cpu(
    torch::Tensor query_points,  // [N', 3]
    torch::Tensor query_width,   // [N',]
    torch::Tensor points,        // [N, 3]
    torch::Tensor point_attrs,   // [N, C]
    torch::Tensor node2point_index,
    torch::Tensor node2point_indexstart,
    torch::Tensor node_children_list,
    torch::Tensor node_attrs,
    torch::Tensor node_is_leaf_list,
    torch::Tensor node_half_w_list,
    torch::Tensor node_reppoints,
    torch::Tensor num_points_in_node
);


torch::Tensor multiply_by_AT_cpu(
    torch::Tensor query_points,  // [N', 3]
    torch::Tensor query_width,   // [N',]
    torch::Tensor points,        // [N, 3]
    torch::Tensor point_attrs,   // [N, C]
    torch::Tensor node2point_index,
    torch::Tensor node2point_indexstart,
    torch::Tensor node_children_list,
    torch::Tensor node_attrs,
    torch::Tensor node_is_leaf_list,
    torch::Tensor node_half_w_list,
    torch::Tensor node_reppoints,
    torch::Tensor num_points_in_node
);

torch::Tensor multiply_by_G_cpu(
    torch::Tensor query_points,  // [N', 3]
    torch::Tensor query_width,   // [N',]
    torch::Tensor points,        // [N, 3]
    torch::Tensor point_attrs,   // [N, C]
    torch::Tensor node2point_index,
    torch::Tensor node2point_indexstart,
    torch::Tensor node_children_list,
    torch::Tensor node_attrs,
    torch::Tensor node_is_leaf_list,
    torch::Tensor node_half_w_list,
    torch::Tensor node_reppoints,
    torch::Tensor num_points_in_node
);

// std::vector<torch::Tensor> scatter_points_to_nodes(torch::Tensor grad_h);
// std::vector<torch::Tensor> multiply_A(torch::Tensor grad_h);
// std::vector<torch::Tensor> multiply_AT(torch::Tensor grad_h);

