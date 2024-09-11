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

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include "wn_treecode_cuda.h"



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

    CHECK_INPUT_FOR_CUDA(node_parent_list);
    CHECK_INPUT_FOR_CUDA(node_children_list);
    CHECK_INPUT_FOR_CUDA(points);
    CHECK_INPUT_FOR_CUDA(point_weights);
    CHECK_INPUT_FOR_CUDA(point_attrs);
    CHECK_INPUT_FOR_CUDA(node2point_index);
    CHECK_INPUT_FOR_CUDA(node2point_indexstart);
    CHECK_INPUT_FOR_CUDA(num_points_in_node);
    CHECK_INPUT_FOR_CUDA(node_is_leaf_list);

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

    signedindex_t num_blocks = (num_nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    AT_DISPATCH_FLOATING_TYPES(points.type(), "scatter_point_attrs_to_nodes_leaf_cuda_kernel", ([&] {
        scatter_point_attrs_to_nodes_leaf_cuda_kernel<scalar_t><<<num_blocks, THREADS_PER_BLOCK>>>(
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
        AT_DISPATCH_FLOATING_TYPES(points.type(), "find_next_to_scatter_cuda_kernel", ([&] {
            find_next_to_scatter_cuda_kernel<scalar_t><<<num_blocks, THREADS_PER_BLOCK>>>(
                node_children_list.data<signedindex_t>(),
                node_is_leaf_list.data<bool>(),
                scattered_mask.data<bool>(),
                next_to_scatter_mask.data<bool>(),
                node2point_index.data<signedindex_t>(),
                num_nodes
            );
        }));

        AT_DISPATCH_FLOATING_TYPES(points.type(), "scatter_point_attrs_to_nodes_nonleaf_cuda_kernel", ([&] {
            scatter_point_attrs_to_nodes_nonleaf_cuda_kernel<scalar_t><<<num_blocks, THREADS_PER_BLOCK>>>(
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

    CHECK_INPUT_FOR_CUDA(query_points);
    CHECK_INPUT_FOR_CUDA(query_width);
    CHECK_INPUT_FOR_CUDA(points);
    CHECK_INPUT_FOR_CUDA(point_attrs);
    CHECK_INPUT_FOR_CUDA(node2point_index);
    CHECK_INPUT_FOR_CUDA(node2point_indexstart);
    CHECK_INPUT_FOR_CUDA(node_children_list);
    CHECK_INPUT_FOR_CUDA(node_attrs);
    CHECK_INPUT_FOR_CUDA(node_is_leaf_list);
    CHECK_INPUT_FOR_CUDA(node_half_w_list);
    CHECK_INPUT_FOR_CUDA(node_reppoints);
    CHECK_INPUT_FOR_CUDA(num_points_in_node);

    signedindex_t num_queries = query_points.size(0);

    auto float_tensor_options = torch::TensorOptions().dtype(points.dtype()).device(points.device());
    auto out_attrs = torch::zeros({query_points.size(0), 1}, float_tensor_options);

    signedindex_t num_blocks = (num_queries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    AT_DISPATCH_FLOATING_TYPES(points.type(), "multiply_by_A_cuda_kernel", ([&] {
        multiply_by_A_cuda_kernel<scalar_t><<<num_blocks, THREADS_PER_BLOCK>>>(
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

    CHECK_INPUT_FOR_CUDA(query_points);
    CHECK_INPUT_FOR_CUDA(query_width);
    CHECK_INPUT_FOR_CUDA(points);
    CHECK_INPUT_FOR_CUDA(point_attrs);
    CHECK_INPUT_FOR_CUDA(node2point_index);
    CHECK_INPUT_FOR_CUDA(node2point_indexstart);
    CHECK_INPUT_FOR_CUDA(node_children_list);
    CHECK_INPUT_FOR_CUDA(node_attrs);
    CHECK_INPUT_FOR_CUDA(node_is_leaf_list);
    CHECK_INPUT_FOR_CUDA(node_half_w_list);
    CHECK_INPUT_FOR_CUDA(node_reppoints);
    CHECK_INPUT_FOR_CUDA(num_points_in_node);

    signedindex_t num_queries = query_points.size(0);

    auto float_tensor_options = torch::TensorOptions().dtype(points.dtype()).device(points.device());
    // auto float_tensor_options = torch::TensorOptions().dtype(points.dtype()).device(torch::kCUDA);
    auto out_attrs = torch::zeros({query_points.size(0), SPATIAL_DIM}, float_tensor_options);
    // auto out_attrs = torch::zeros({query_points.size(0), SPATIAL_DIM});

    // std::cout << "[DEBUG] created AT result\n";

    signedindex_t num_blocks = (num_queries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    AT_DISPATCH_FLOATING_TYPES(points.type(), "multiply_by_AT_cuda_kernel", ([&] {
        multiply_by_AT_cuda_kernel<scalar_t><<<num_blocks, THREADS_PER_BLOCK>>>(
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
    
    CHECK_INPUT_FOR_CUDA(query_points);
    CHECK_INPUT_FOR_CUDA(query_width);
    CHECK_INPUT_FOR_CUDA(points);
    CHECK_INPUT_FOR_CUDA(point_attrs);
    CHECK_INPUT_FOR_CUDA(node2point_index);
    CHECK_INPUT_FOR_CUDA(node2point_indexstart);
    CHECK_INPUT_FOR_CUDA(node_children_list);
    CHECK_INPUT_FOR_CUDA(node_attrs);
    CHECK_INPUT_FOR_CUDA(node_is_leaf_list);
    CHECK_INPUT_FOR_CUDA(node_half_w_list);
    CHECK_INPUT_FOR_CUDA(node_reppoints);
    CHECK_INPUT_FOR_CUDA(num_points_in_node);

    signedindex_t num_queries = query_points.size(0);

    auto float_tensor_options = torch::TensorOptions().dtype(points.dtype()).device(points.device());
    auto out_attrs = torch::zeros({query_points.size(0), SPATIAL_DIM}, float_tensor_options);

    signedindex_t num_blocks = (num_queries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    AT_DISPATCH_FLOATING_TYPES(points.type(), "multiply_by_G_cuda_kernel", ([&] {
        multiply_by_G_cuda_kernel<scalar_t><<<num_blocks, THREADS_PER_BLOCK>>>(
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
  m.def("scatter_point_attrs_to_nodes", &scatter_point_attrs_to_nodes, "scatter_point_attrs_to_nodes (CUDA)");
  m.def("multiply_by_A", &multiply_by_A, "multiply by A (CUDA)");
  m.def("multiply_by_AT", &multiply_by_AT, "multiply by AT (CUDA)");
  m.def("multiply_by_G", &multiply_by_G, "multiply by AT (CUDA)");
}
