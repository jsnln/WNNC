#include <torch/extension.h>
#include "torch_treecode_cuda.h"
#include <vector>
#include <fstream>

#define NUM_OCT_CHILDREN 8
typedef long signedindex_t;


int color_list[11*3] = {
    255, 0, 0,
    0, 255, 0,
    0, 0, 255,
    0, 255, 255,
    255, 0, 255,
    255, 255, 0,
    255, 80, 80,
    80, 255, 80,
    80, 80, 255,
    0, 160, 255,
};

template<typename scalar_t>
struct OctNode {
    scalar_t c_x, c_y, c_z;
    scalar_t half_w;   // half width
    
    std::vector<signedindex_t> point_indices;
    OctNode<scalar_t>* children[NUM_OCT_CHILDREN] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    OctNode<scalar_t>* parent = nullptr;

    scalar_t serial_index = -1;
    scalar_t depth = -1;
    bool is_leaf = false;
};


template<typename scalar_t>
void debug_output_grid_corners_recursive(std::ofstream& fout, const OctNode<scalar_t>* cur_node, signedindex_t depth) {
    if (cur_node == nullptr) {
        return;
    }
    fout << "v " << cur_node->c_x - cur_node->half_w << " " << cur_node->c_y - cur_node->half_w << " " << cur_node->c_z - cur_node->half_w << " " << color_list[3*depth+0] << " " << color_list[3*depth+1] << " " << color_list[3*depth+2] << "\n"
         << "v " << cur_node->c_x + cur_node->half_w << " " << cur_node->c_y - cur_node->half_w << " " << cur_node->c_z - cur_node->half_w << " " << color_list[3*depth+0] << " " << color_list[3*depth+1] << " " << color_list[3*depth+2] << "\n"
         << "v " << cur_node->c_x - cur_node->half_w << " " << cur_node->c_y + cur_node->half_w << " " << cur_node->c_z - cur_node->half_w << " " << color_list[3*depth+0] << " " << color_list[3*depth+1] << " " << color_list[3*depth+2] << "\n"
         << "v " << cur_node->c_x + cur_node->half_w << " " << cur_node->c_y + cur_node->half_w << " " << cur_node->c_z - cur_node->half_w << " " << color_list[3*depth+0] << " " << color_list[3*depth+1] << " " << color_list[3*depth+2] << "\n"
         << "v " << cur_node->c_x - cur_node->half_w << " " << cur_node->c_y - cur_node->half_w << " " << cur_node->c_z + cur_node->half_w << " " << color_list[3*depth+0] << " " << color_list[3*depth+1] << " " << color_list[3*depth+2] << "\n"
         << "v " << cur_node->c_x + cur_node->half_w << " " << cur_node->c_y - cur_node->half_w << " " << cur_node->c_z + cur_node->half_w << " " << color_list[3*depth+0] << " " << color_list[3*depth+1] << " " << color_list[3*depth+2] << "\n"
         << "v " << cur_node->c_x - cur_node->half_w << " " << cur_node->c_y + cur_node->half_w << " " << cur_node->c_z + cur_node->half_w << " " << color_list[3*depth+0] << " " << color_list[3*depth+1] << " " << color_list[3*depth+2] << "\n"
         << "v " << cur_node->c_x + cur_node->half_w << " " << cur_node->c_y + cur_node->half_w << " " << cur_node->c_z + cur_node->half_w << " " << color_list[3*depth+0] << " " << color_list[3*depth+1] << " " << color_list[3*depth+2] << "\n";
    
    for (signedindex_t k = 0; k < NUM_OCT_CHILDREN; k++) {
        if (cur_node->children[k] != nullptr) {
            debug_output_grid_corners_recursive(fout, cur_node->children[k], depth+1);
        }
    }
}

template<typename scalar_t>
void debug_output_grid_corners(std::string filename, const OctNode<scalar_t>* root) {
    std::ofstream fout(filename);
    debug_output_grid_corners_recursive<scalar_t>(fout, root, 0);
    fout.close();
}


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
    ) {

    signedindex_t num_points = point_indices.size();
    if (num_points == 0) {
        return nullptr;
    }

    // basic properties of this node
    OctNode<scalar_t>* node = new OctNode<scalar_t>;
    node->serial_index = cur_node_index;
    cur_node_index += 1;

    node->c_x = c_x;
    node->c_y = c_y;
    node->c_z = c_z;
    node->half_w = half_w;
    node->point_indices = point_indices;
    node->parent = parent;
    node->depth = cur_depth;

    // stop splitting if maximum depth reached, or point threshold reached
    if ((max_depth >= 0 && cur_depth >= max_depth) || num_points <= max_points_per_node) {
        node->is_leaf = true;
        return node;
    }

    std::vector<signedindex_t> subdivision_point_indices[NUM_OCT_CHILDREN];

    for (signedindex_t i = 0; i < num_points; i++) {
        signedindex_t pid = point_indices[i];
        scalar_t x = point_coords[3*pid+0];
        scalar_t y = point_coords[3*pid+1];
        scalar_t z = point_coords[3*pid+2];

        // morton code
        signedindex_t child_node_code = 0;
        if (x >= c_x) child_node_code += 1;
        if (y >= c_y) child_node_code += 2;
        if (z >= c_z) child_node_code += 4;

        subdivision_point_indices[child_node_code].push_back(pid);
    }

    scalar_t next_w = half_w / 2.0;   // width for next level
    if (subdivision_point_indices[0].size() > 0)
        node->children[0] = build_tree_cpu_recursive(point_coords, subdivision_point_indices[0], node,
            c_x-next_w, c_y-next_w, c_z-next_w, next_w, cur_depth+1, cur_node_index, max_depth, max_points_per_node);
    if (subdivision_point_indices[1].size() > 0)
        node->children[1] = build_tree_cpu_recursive(point_coords, subdivision_point_indices[1], node,
            c_x+next_w, c_y-next_w, c_z-next_w, next_w, cur_depth+1, cur_node_index, max_depth, max_points_per_node);
    if (subdivision_point_indices[2].size() > 0)
        node->children[2] = build_tree_cpu_recursive(point_coords, subdivision_point_indices[2], node,
            c_x-next_w, c_y+next_w, c_z-next_w, next_w, cur_depth+1, cur_node_index, max_depth, max_points_per_node);
    if (subdivision_point_indices[3].size() > 0)
        node->children[3] = build_tree_cpu_recursive(point_coords, subdivision_point_indices[3], node,
            c_x+next_w, c_y+next_w, c_z-next_w, next_w, cur_depth+1, cur_node_index, max_depth, max_points_per_node);
    if (subdivision_point_indices[4].size() > 0)
        node->children[4] = build_tree_cpu_recursive(point_coords, subdivision_point_indices[4], node,
            c_x-next_w, c_y-next_w, c_z+next_w, next_w, cur_depth+1, cur_node_index, max_depth, max_points_per_node);
    if (subdivision_point_indices[5].size() > 0)
        node->children[5] = build_tree_cpu_recursive(point_coords, subdivision_point_indices[5], node,
            c_x+next_w, c_y-next_w, c_z+next_w, next_w, cur_depth+1, cur_node_index, max_depth, max_points_per_node);
    if (subdivision_point_indices[6].size() > 0)
        node->children[6] = build_tree_cpu_recursive(point_coords, subdivision_point_indices[6], node,
            c_x-next_w, c_y+next_w, c_z+next_w, next_w, cur_depth+1, cur_node_index, max_depth, max_points_per_node);
    if (subdivision_point_indices[7].size() > 0)
        node->children[7] = build_tree_cpu_recursive(point_coords, subdivision_point_indices[7], node,
            c_x+next_w, c_y+next_w, c_z+next_w, next_w, cur_depth+1, cur_node_index, max_depth, max_points_per_node);
    bool is_leaf = true;
    for (signedindex_t k = 0; k < NUM_OCT_CHILDREN; k++) {
        if (node->children[k] != nullptr) {
            is_leaf = false;
        }
    }
    node->is_leaf = is_leaf;
    return node;
}


template<typename scalar_t>
void compute_tree_attributes(
        const OctNode<scalar_t> * cur_node,
        signedindex_t & num_nodes,
        signedindex_t & num_leaves,
        signedindex_t & depth
    ) {
    if (cur_node != nullptr) {
        num_nodes += 1;
        if (cur_node->depth > depth) {
            depth = cur_node->depth;
        }
        if (cur_node->is_leaf) {
            num_leaves++;
        }
        for (signedindex_t i = 0; i < NUM_OCT_CHILDREN; i++) {
            compute_tree_attributes(cur_node->children[i], num_nodes, num_leaves, depth);
        }
        
    }
}

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
    ) {
    if (cur_node == nullptr) {
        return;
    }

    signedindex_t node_index = cur_node->serial_index;


    signedindex_t node2point_indexstart = stdvec_node2point_index.size();
    ptr_node_half_w_list[node_index] = cur_node->half_w;

    // set up point2node indexing and vice versa
    ptr_num_points_in_node[node_index] = cur_node->point_indices.size();
    ptr_node2point_indexstart[node_index] = node2point_indexstart;
    for (signedindex_t i = 0; i < cur_node->point_indices.size(); i++) {
        // point2node
        // point2node_index[cur_node->depth][pid] = cur_node->serialize_index;

        // node2point
        signedindex_t point_index = cur_node->point_indices[i];
        stdvec_node2point_index.push_back(point_index);
    }

    // actual serialization
    if (cur_node->parent == nullptr) {
        ptr_node_parent_list[node_index] = -1;    
    } else {
        ptr_node_parent_list[node_index] = cur_node->parent->serial_index;
    }
    ptr_node_is_leaf_list[node_index] = cur_node->is_leaf;
    
    for (signedindex_t k = 0; k < NUM_OCT_CHILDREN; k++) {
        if (cur_node->children[k] != nullptr) {
            ptr_node_children_list[node_index*NUM_OCT_CHILDREN + k] = cur_node->children[k]->serial_index;
            serialize_tree_recursive(cur_node->children[k],
                                     ptr_node_parent_list,
                                     ptr_node_children_list,
                                     ptr_node_is_leaf_list,
                                     ptr_node_half_w_list,
                                     ptr_num_points_in_node,
                                     ptr_node2point_indexstart,
                                     stdvec_node2point_index);
        } else {
            ptr_node_children_list[node_index*NUM_OCT_CHILDREN + k] = -1;
        }
    }
}

template<typename scalar_t>
void free_tree_recursive(OctNode<scalar_t>* cur_node) {
    if (cur_node == nullptr) {
        return;
    }
    for (signedindex_t k = 0; k < NUM_OCT_CHILDREN; k++) {
        free_tree_recursive(cur_node->children[k]);
    }
    delete cur_node;
}

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



// C++ interface
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
    CHECK_INPUT_FOR_CUDA(node_parent_list);
    CHECK_INPUT_FOR_CUDA(node_children_list);
    CHECK_INPUT_FOR_CUDA(points);
    CHECK_INPUT_FOR_CUDA(point_weights);
    CHECK_INPUT_FOR_CUDA(point_attrs);
    CHECK_INPUT_FOR_CUDA(node2point_index);
    CHECK_INPUT_FOR_CUDA(node2point_indexstart);
    CHECK_INPUT_FOR_CUDA(num_points_in_node);
    CHECK_INPUT_FOR_CUDA(node_is_leaf_list);

    return scatter_point_attrs_to_nodes_cuda(
        node_parent_list,
        node_children_list,
        points,
        point_weights,
        point_attrs,
        node2point_index,
        node2point_indexstart,
        num_points_in_node,
        node_is_leaf_list,
        tree_depth);
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
    
    return multiply_by_A_cuda(query_points,  // [N', 3]
        query_width,   // [N',]
        points,        // [N, 3]
        point_attrs,   // [N, C]
        node2point_index,
        node2point_indexstart,
        node_children_list,
        node_attrs,
        node_is_leaf_list,
        node_half_w_list,
        node_reppoints,
        num_points_in_node);
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
    
    return multiply_by_G_cuda(query_points,  // [N', 3]
        query_width,   // [N',]
        points,        // [N, 3]
        point_attrs,   // [N, C]
        node2point_index,
        node2point_indexstart,
        node_children_list,
        node_attrs,
        node_is_leaf_list,
        node_half_w_list,
        node_reppoints,
        num_points_in_node);
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

    // std::cout << "[DEBUG] Entered multiply_by_AT\n";
    
    return multiply_by_AT_cuda(
        query_points,  // [N', 3]
        query_width,   // [N',]
        points,        // [N, 3]
        point_attrs,   // [N, C]
        node2point_index,
        node2point_indexstart,
        node_children_list,
        node_attrs,
        node_is_leaf_list,
        node_half_w_list,
        node_reppoints,
        num_points_in_node);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_tree", &build_tree, "build tree (CPU)");
  m.def("scatter_point_attrs_to_nodes", &scatter_point_attrs_to_nodes, "scatter_point_attrs_to_nodes (CUDA)");
  m.def("multiply_by_A", &multiply_by_A, "multiply by A (CUDA)");
  m.def("multiply_by_AT", &multiply_by_AT, "multiply by AT (CUDA)");
  m.def("multiply_by_G", &multiply_by_G, "multiply by AT (CUDA)");
//   m.def("multiply_by_A_cuda_debug_single_thread", &multiply_by_A_cuda_debug_single_thread, "multiply by A (CUDA)");
}



/// @debug for debug
// template<typename scalar_t>
// scalar_t get_point2point_dist2_cpu(const scalar_t* point1, const scalar_t* point2) {
//     // squared distance between 2 points, both dim == 3
//     // user is responsible for making sure dim == 3
//     return std::pow(point1[0] - point2[0], scalar_t(2.0)) +
//            std::pow(point1[1] - point2[1], scalar_t(2.0)) +
//            std::pow(point1[2] - point2[2], scalar_t(2.0));
// }

// template<typename scalar_t>
// void multiply_by_A_cuda_kernel_debug_single_thread(
//         const scalar_t* query_points,  // [N', 3]
//         const scalar_t* query_width,   // [N',]
//         const scalar_t* points,        // [N, 3]
//         const scalar_t* point_attrs,   // [N, C]
//         const signedindex_t* node2point_index,
//         const signedindex_t* node2point_indexstart,
//         const signedindex_t* node_children_list,
//         const scalar_t* node_attrs,
//         const bool* node_is_leaf_list,
//         const scalar_t* node_half_w_list,
//         const scalar_t* node_reppoints,
//         const signedindex_t* num_points_in_node,
//         scalar_t* out_attrs,           // [N, 3]
//         signedindex_t num_queries,
//         signedindex_t query_index
//     ) {
//     // the caller is responsible for making sure 'point_attrs' is [N, C=3]
//     // signedindex_t query_index = blockDim.x * blockIdx.x + threadIdx.x;
//     if (query_index < num_queries) {
// 
//         scalar_t out_val = 0.0;
//  
//         constexpr signedindex_t search_stack_max_size = ALLOWED_MAX_DEPTH*(NUM_OCT_CHILDREN - 1) + 1;
//         signedindex_t search_stack[search_stack_max_size] = {};
//         signedindex_t search_stack_top = 0;
// 
//         // a push
//         search_stack[search_stack_top++] = 0;
// 
//         signedindex_t DEBUG_compute_counter = 0;
// 
//         while (search_stack_top > 0) {
//             assert(search_stack_top < search_stack_max_size);
//  
//             // a pop
//             signedindex_t cur_node_index = search_stack[--search_stack_top];
//             scalar_t point2node_dist2 = get_point2point_dist2_cpu(query_points + query_index*SPATIAL_DIM,
//                                                               node_reppoints + cur_node_index*SPATIAL_DIM);
// 
//             /// @case 1: the query point is far from the sample,
//             //           approximate the query value with the node center
//             if (point2node_dist2 > pow(scalar_t(TREECODE_THRESHOLD * 2.0f) * node_half_w_list[cur_node_index], scalar_t(2.0f))) {
//             // if (false) {
//                 DEBUG_compute_counter++;
// 
//                 scalar_t diff[SPATIAL_DIM];     // x - y
//                 scalar_t dist, dist2, dist3;    // d, d^2, d^3
//                 dist = dist2 = dist3 = 0.0;
// 
//                 for (signedindex_t d = 0; d < SPATIAL_DIM; d++) {
//                     diff[d] = query_points[query_index*SPATIAL_DIM + d] - node_reppoints[cur_node_index*SPATIAL_DIM + d];
//                     dist2 += (diff[d] * diff[d]);
//                 }
//                 dist = sqrt(dist2);
// 
//                 if (dist >= query_width[query_index]) {
//                     // outside of width smoothing range
//                     dist3 = dist * dist2;
//                     for (signedindex_t d = 0; d < SPATIAL_DIM; d++) {
//                         out_val += (-1 * node_attrs[cur_node_index * SPATIAL_DIM + d] * diff[d]) / dist3;
//                     }
//                     // std::cout << "node rep used, out_val: " << out_val << "\n";
//                 } else {
//                     // outside of width smoothing range
//                 }
//             }
//             /// @case 2: the query point is not that far,
//             //           check if this node has any children, if any, push them to the search stack
//             else {
// 
//                 // if nonleaf, push children to the search stack
//                 if (!node_is_leaf_list[cur_node_index]) {
//                     for (signedindex_t k = 0; k < NUM_OCT_CHILDREN; k++) {
//                         if (node_children_list[cur_node_index * NUM_OCT_CHILDREN + k] != -1) {
// 
//                             search_stack[search_stack_top++] = node_children_list[cur_node_index * NUM_OCT_CHILDREN + k];
//                             assert(search_stack_top < search_stack_max_size);
// 
//                             std::cout << "push: " << cur_node_index << ", stack: " << search_stack_top << " / " << search_stack_max_size << "\n";
//                         }
//                     }
//                 } else {  /// @case 3: this node is a leaf node, compute over samples
//                     for (signedindex_t k = 0; k < num_points_in_node[cur_node_index]; k++) {
//                         DEBUG_compute_counter++;
//                         signedindex_t point_index = node2point_index[node2point_indexstart[cur_node_index] + k];
// 
//                         scalar_t diff[SPATIAL_DIM];     // x - y
//                         scalar_t dist, dist2, dist3;    // d, d^2, d^3
//                         dist = dist2 = dist3 = 0.0;
// 
//                         for (signedindex_t d = 0; d < SPATIAL_DIM; d++) {
//                             diff[d] = query_points[query_index*SPATIAL_DIM + d] - points[point_index*SPATIAL_DIM + d];
//                             dist2 += (diff[d] * diff[d]);
//                         }
//                         dist = sqrt(dist2);
// 
// 
//                         if (dist >= query_width[query_index]) {
//                             // outside of width smoothing range
//                             dist3 = dist * dist2;
//                             for (signedindex_t d = 0; d < SPATIAL_DIM; d++) {
//                                 out_val += (-1 * point_attrs[point_index * SPATIAL_DIM + d] * diff[d]) / dist3;
//                             }
//                         } else {
// 
//                         }
//                     }
//                 }
//             }
//         }
//         out_attrs[query_index] = out_val;
//     }
// }



// torch::Tensor multiply_by_A_cuda_debug_single_thread(
//         torch::Tensor query_points,  // [N', 3]
//         torch::Tensor query_width,   // [N',]
//         torch::Tensor points,        // [N, 3]
//         torch::Tensor point_attrs,   // [N, C]
//         torch::Tensor node2point_index,
//         torch::Tensor node2point_indexstart,
//         torch::Tensor node_children_list,
//         torch::Tensor node_attrs,
//         torch::Tensor node_is_leaf_list,
//         torch::Tensor node_half_w_list,
//         torch::Tensor node_reppoints,
//         torch::Tensor num_points_in_node
//         ) {
// 
//     signedindex_t num_queries = query_points.size(0);
// 
//     auto float_tensor_options = torch::TensorOptions().dtype(points.dtype()).device(points.device());
//     auto out_attrs = torch::zeros({query_points.size(0), 1}, float_tensor_options);
// 
//     signedindex_t num_blocks = (num_queries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//  
//     multiply_by_A_cuda_kernel_debug_single_thread<double>(
//             query_points.data<double>(),  // [N', 3]
//             query_width.data<double>(),   // [N',]
//             points.data<double>(),        // [N, 3]
//             point_attrs.data<double>(),   // [N, C]
//             node2point_index.data<signedindex_t>(),
//             node2point_indexstart.data<signedindex_t>(),
//             node_children_list.data<signedindex_t>(),
//             node_attrs.data<double>(),
//             node_is_leaf_list.data<bool>(),
//             node_half_w_list.data<double>(),
//             node_reppoints.data<double>(),
//             num_points_in_node.data<signedindex_t>(),
//             out_attrs.data<double>(),           // [N, 3]
//             num_queries,
//             0
//         );

//     return out_attrs;
// }



