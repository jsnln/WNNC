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


#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_FOR_CUDA(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_FOR_CPU(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

#include "wn_treecode_cpu.h"
#include <vector>
#include <fstream>

#define NUM_OCT_CHILDREN 8
typedef long signedindex_t;

template<typename scalar_t>
std::vector<torch::Tensor> build_tree_cpu(torch::Tensor points_tensor, signedindex_t max_depth) {

    const auto num_points = points_tensor.size(0);
    std::vector<signedindex_t> point_indices(num_points);
    for (signedindex_t i = 0; i < num_points; i++) {
        point_indices[i] = i;
    }

    signedindex_t cur_node_index = 0;
    auto root = build_tree_cpu_recursive<scalar_t>(
        points_tensor.data<scalar_t>(),
        point_indices,
        /*parent = */nullptr,
        /*c_x, c_y, c_z = */0.0, 0.0, 0.0,
        /*half_width = */1.0,
        /*depth = */0,
        /*cur_node_index = */cur_node_index,
        /*max_depth = */max_depth,
        /*max_points_per_node*/1
    );

    signedindex_t num_nodes = 0;
    signedindex_t num_leaves = 0;
    signedindex_t tree_depth = 0;
    compute_tree_attributes<scalar_t>(root, num_nodes, num_leaves, tree_depth);
    std::cout << "num_nodes: " << num_nodes << ", num_leaves: " << num_leaves << ", tree depth: " << tree_depth << "\n";

    // debug_output_grid_corners("debug_grid.obj", root);

    auto long_tensor_options = torch::TensorOptions().dtype(torch::kLong);
    auto node_parent_list = torch::zeros({num_nodes}, long_tensor_options);
    auto node_children_list = torch::zeros({num_nodes, NUM_OCT_CHILDREN}, long_tensor_options);
    
    auto bool_tensor_options = torch::TensorOptions().dtype(torch::kBool);
    auto node_is_leaf_list = torch::zeros({num_nodes}, bool_tensor_options);

    auto num_points_in_node = torch::zeros({num_nodes}, long_tensor_options);
    auto node2point_indexstart = torch::zeros({num_nodes}, long_tensor_options);
    std::vector<signedindex_t> stdvec_node2point_index;

    auto float_tensor_options = torch::TensorOptions().dtype(points_tensor.dtype());
    auto node_half_w_list = torch::zeros({num_nodes}, float_tensor_options);

    serialize_tree_recursive(root,
                             node_parent_list.data<signedindex_t>(),
                             node_children_list.data<signedindex_t>(),
                             node_is_leaf_list.data<bool>(),
                             node_half_w_list.data<scalar_t>(),
                             num_points_in_node.data<signedindex_t>(),
                             node2point_indexstart.data<signedindex_t>(),
                             stdvec_node2point_index);

    auto node2point_index = torch::zeros({stdvec_node2point_index.size()}, long_tensor_options);
    std::memcpy(node2point_index.data<signedindex_t>(), stdvec_node2point_index.data(), stdvec_node2point_index.size()*sizeof(signedindex_t));

    std::cout << "tree_depth*num_points: " << tree_depth*num_points << ", stdvec_node2point_index.size(): " << stdvec_node2point_index.size() << "\n";

    free_tree_recursive(root);

    return {node_parent_list, node_children_list, node_is_leaf_list, node_half_w_list, num_points_in_node, node2point_index, node2point_indexstart};
}

std::vector<torch::Tensor> build_tree(torch::Tensor points_tensor, signedindex_t max_depth) {
    CHECK_INPUT_FOR_CPU(points_tensor);

    // similar to  AT_DISPATCH_FLOATING_TYPES
    const auto & _st = ::detail::scalar_type(points_tensor.type());
    // RECORD_KERNEL_FUNCTION_DTYPE("build_tree", _st);     // what does this do?
    switch (_st) {
        case torch::ScalarType::Double:
            return build_tree_cpu<double>(points_tensor, max_depth);
        case torch::ScalarType::Float:
            return build_tree_cpu<float>(points_tensor, max_depth);
        default:
            AT_ERROR("build_tree", " not implemented for dtype '", toString(_st), "'");
    }
}

std::vector<torch::Tensor> scatter_point_attrs_to_nodes(
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
        ) {
    
    CHECK_INPUT_FOR_CPU(node_parent_list);
    CHECK_INPUT_FOR_CPU(node_children_list);
    CHECK_INPUT_FOR_CPU(points);
    CHECK_INPUT_FOR_CPU(point_weights);
    CHECK_INPUT_FOR_CPU(point_attrs);
    CHECK_INPUT_FOR_CPU(node2point_index);
    CHECK_INPUT_FOR_CPU(node2point_indexstart);
    CHECK_INPUT_FOR_CPU(num_points_in_node);
    CHECK_INPUT_FOR_CPU(node_is_leaf_list);

    signedindex_t num_nodes = node_parent_list.size(0);
    signedindex_t attr_dim = point_attrs.size(1);
    assert(attr_dim == SPATIAL_DIM or attr_dim == 1);

    auto bool_tensor_options = torch::TensorOptions().dtype(torch::kBool).device(points.device());
    auto scattered_mask = torch::zeros({num_nodes}, bool_tensor_options);
    auto next_to_scatter_mask = torch::zeros({num_nodes}, bool_tensor_options);
    
    auto float_tensor_options = torch::TensorOptions().dtype(points.dtype()).device(points.device());
    auto out_node_attrs = torch::zeros({num_nodes, point_attrs.size(1)}, float_tensor_options);
    auto out_node_reppoints = torch::zeros({num_nodes, SPATIAL_DIM}, float_tensor_options);
    auto out_node_weights = torch::zeros({num_nodes}, float_tensor_options);

    AT_DISPATCH_FLOATING_TYPES(points.type(), "scatter_point_attrs_to_nodes_leaf_cpu_kernel_launcher", ([&] {
        scatter_point_attrs_to_nodes_leaf_cpu_kernel_launcher<scalar_t>(
            node_parent_list.data<signedindex_t>(),
            points.data<scalar_t>(),
            point_weights.data<scalar_t>(),
            point_attrs.data<scalar_t>(),
            node2point_index.data<signedindex_t>(),
            node2point_indexstart.data<signedindex_t>(),
            num_points_in_node.data<signedindex_t>(),
            node_is_leaf_list.data<bool>(),
            scattered_mask.data<bool>(),
            out_node_attrs.data<scalar_t>(),
            out_node_reppoints.data<scalar_t>(),
            out_node_weights.data<scalar_t>(),
            attr_dim,
            num_nodes
            );
    }));

    for (signedindex_t depth = tree_depth-1; depth >= 0; depth--) {
        AT_DISPATCH_FLOATING_TYPES(points.type(), "find_next_to_scatter_cpu_kernel_launcher", ([&] {
            find_next_to_scatter_cpu_kernel_launcher<scalar_t>(
                node_children_list.data<signedindex_t>(),
                node_is_leaf_list.data<bool>(),
                scattered_mask.data<bool>(),
                next_to_scatter_mask.data<bool>(),
                node2point_index.data<signedindex_t>(),
                num_nodes
            );
        }));

        AT_DISPATCH_FLOATING_TYPES(points.type(), "scatter_point_attrs_to_nodes_nonleaf_cpu_kernel_launcher", ([&] {
            scatter_point_attrs_to_nodes_nonleaf_cpu_kernel_launcher<scalar_t>(
                node_parent_list.data<signedindex_t>(),
                node_children_list.data<signedindex_t>(),
                points.data<scalar_t>(),
                point_weights.data<scalar_t>(),
                point_attrs.data<scalar_t>(),
                node2point_index.data<signedindex_t>(),
                node2point_indexstart.data<signedindex_t>(),
                num_points_in_node.data<signedindex_t>(),
                node_is_leaf_list.data<bool>(),
                scattered_mask.data<bool>(),
                next_to_scatter_mask.data<bool>(),
                out_node_attrs.data<scalar_t>(),
                out_node_reppoints.data<scalar_t>(),
                out_node_weights.data<scalar_t>(),
                attr_dim,
                num_nodes
            );
        }));
    }

    return {out_node_attrs, out_node_reppoints, out_node_weights};
}


torch::Tensor multiply_by_A(
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
        ) {

    CHECK_INPUT_FOR_CPU(query_points);
    CHECK_INPUT_FOR_CPU(query_width);
    CHECK_INPUT_FOR_CPU(points);
    CHECK_INPUT_FOR_CPU(point_attrs);
    CHECK_INPUT_FOR_CPU(node2point_index);
    CHECK_INPUT_FOR_CPU(node2point_indexstart);
    CHECK_INPUT_FOR_CPU(node_children_list);
    CHECK_INPUT_FOR_CPU(node_attrs);
    CHECK_INPUT_FOR_CPU(node_is_leaf_list);
    CHECK_INPUT_FOR_CPU(node_half_w_list);
    CHECK_INPUT_FOR_CPU(node_reppoints);
    CHECK_INPUT_FOR_CPU(num_points_in_node);

    signedindex_t num_queries = query_points.size(0);

    auto float_tensor_options = torch::TensorOptions().dtype(points.dtype()).device(points.device());
    auto out_attrs = torch::zeros({query_points.size(0), 1}, float_tensor_options);

    AT_DISPATCH_FLOATING_TYPES(points.type(), "multiply_by_A_cpu_kernel_launcher", ([&] {
        multiply_by_A_cpu_kernel_launcher<scalar_t>(
            query_points.data<scalar_t>(),  // [N', 3]
            query_width.data<scalar_t>(),   // [N',]
            points.data<scalar_t>(),        // [N, 3]
            point_attrs.data<scalar_t>(),   // [N, C]
            node2point_index.data<signedindex_t>(),
            node2point_indexstart.data<signedindex_t>(),
            node_children_list.data<signedindex_t>(),
            node_attrs.data<scalar_t>(),
            node_is_leaf_list.data<bool>(),
            node_half_w_list.data<scalar_t>(),
            node_reppoints.data<scalar_t>(),
            num_points_in_node.data<signedindex_t>(),
            out_attrs.data<scalar_t>(),           // [N, 3]
            num_queries
        );
    }));

    return out_attrs;
}


torch::Tensor multiply_by_AT(
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
        ) {
    
    CHECK_INPUT_FOR_CPU(query_points);
    CHECK_INPUT_FOR_CPU(query_width);
    CHECK_INPUT_FOR_CPU(points);
    CHECK_INPUT_FOR_CPU(point_attrs);
    CHECK_INPUT_FOR_CPU(node2point_index);
    CHECK_INPUT_FOR_CPU(node2point_indexstart);
    CHECK_INPUT_FOR_CPU(node_children_list);
    CHECK_INPUT_FOR_CPU(node_attrs);
    CHECK_INPUT_FOR_CPU(node_is_leaf_list);
    CHECK_INPUT_FOR_CPU(node_half_w_list);
    CHECK_INPUT_FOR_CPU(node_reppoints);
    CHECK_INPUT_FOR_CPU(num_points_in_node);

    signedindex_t num_queries = query_points.size(0);

    auto float_tensor_options = torch::TensorOptions().dtype(points.dtype()).device(points.device());
    // auto float_tensor_options = torch::TensorOptions().dtype(points.dtype()).device(torch::kCUDA);
    auto out_attrs = torch::zeros({query_points.size(0), SPATIAL_DIM}, float_tensor_options);
    // auto out_attrs = torch::zeros({query_points.size(0), SPATIAL_DIM});

    // std::cout << "[DEBUG] created AT result\n";

    AT_DISPATCH_FLOATING_TYPES(points.type(), "multiply_by_AT_cpu_kernel_launcher", ([&] {
        multiply_by_AT_cpu_kernel_launcher<scalar_t>(
            query_points.data<scalar_t>(),  // [N', 3]
            query_width.data<scalar_t>(),   // [N',]
            points.data<scalar_t>(),        // [N, 3]
            point_attrs.data<scalar_t>(),   // [N, C]
            node2point_index.data<signedindex_t>(),
            node2point_indexstart.data<signedindex_t>(),
            node_children_list.data<signedindex_t>(),
            node_attrs.data<scalar_t>(),
            node_is_leaf_list.data<bool>(),
            node_half_w_list.data<scalar_t>(),
            node_reppoints.data<scalar_t>(),
            num_points_in_node.data<signedindex_t>(),
            out_attrs.data<scalar_t>(),           // [N, 3]
            num_queries
        );
    }));

    return out_attrs;
}
//////////////////////////////


torch::Tensor multiply_by_G(
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
        ) {

    CHECK_INPUT_FOR_CPU(query_points);
    CHECK_INPUT_FOR_CPU(query_width);
    CHECK_INPUT_FOR_CPU(points);
    CHECK_INPUT_FOR_CPU(point_attrs);
    CHECK_INPUT_FOR_CPU(node2point_index);
    CHECK_INPUT_FOR_CPU(node2point_indexstart);
    CHECK_INPUT_FOR_CPU(node_children_list);
    CHECK_INPUT_FOR_CPU(node_attrs);
    CHECK_INPUT_FOR_CPU(node_is_leaf_list);
    CHECK_INPUT_FOR_CPU(node_half_w_list);
    CHECK_INPUT_FOR_CPU(node_reppoints);
    CHECK_INPUT_FOR_CPU(num_points_in_node);

    signedindex_t num_queries = query_points.size(0);

    auto float_tensor_options = torch::TensorOptions().dtype(points.dtype()).device(points.device());
    auto out_attrs = torch::zeros({query_points.size(0), SPATIAL_DIM}, float_tensor_options);

    AT_DISPATCH_FLOATING_TYPES(points.type(), "multiply_by_A_cpu_kernel_launcher", ([&] {
        multiply_by_G_cpu_kernel_launcher<scalar_t>(
            query_points.data<scalar_t>(),  // [N', 3]
            query_width.data<scalar_t>(),   // [N',]
            points.data<scalar_t>(),        // [N, 3]
            point_attrs.data<scalar_t>(),   // [N, C]
            node2point_index.data<signedindex_t>(),
            node2point_indexstart.data<signedindex_t>(),
            node_children_list.data<signedindex_t>(),
            node_attrs.data<scalar_t>(),
            node_is_leaf_list.data<bool>(),
            node_half_w_list.data<scalar_t>(),
            node_reppoints.data<scalar_t>(),
            num_points_in_node.data<signedindex_t>(),
            out_attrs.data<scalar_t>(),           // [N, 3]
            num_queries
        );
    }));
    return out_attrs;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_tree", &build_tree, "build tree (CPU)");
  m.def("scatter_point_attrs_to_nodes", &scatter_point_attrs_to_nodes, "scatter_point_attrs_to_nodes (CPU)");
  m.def("multiply_by_A", &multiply_by_A, "multiply by A (CPU)");
  m.def("multiply_by_AT", &multiply_by_AT, "multiply by AT (CPU)");
  m.def("multiply_by_G", &multiply_by_G, "multiply by AT (CPU)");
}

