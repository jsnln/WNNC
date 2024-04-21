#include <torch/extension.h>
#include "torch_treecode_cpu.h"
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
    CHECK_INPUT_FOR_CPU(node_parent_list);
    CHECK_INPUT_FOR_CPU(node_children_list);
    CHECK_INPUT_FOR_CPU(points);
    CHECK_INPUT_FOR_CPU(point_weights);
    CHECK_INPUT_FOR_CPU(point_attrs);
    CHECK_INPUT_FOR_CPU(node2point_index);
    CHECK_INPUT_FOR_CPU(node2point_indexstart);
    CHECK_INPUT_FOR_CPU(num_points_in_node);
    CHECK_INPUT_FOR_CPU(node_is_leaf_list);

    return scatter_point_attrs_to_nodes_cpu(
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
    
    return multiply_by_A_cpu(query_points,  // [N', 3]
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
    
    return multiply_by_G_cpu(query_points,  // [N', 3]
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

    // std::cout << "[DEBUG] Entered multiply_by_AT\n";
    
    return multiply_by_AT_cpu(
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
  m.def("scatter_point_attrs_to_nodes", &scatter_point_attrs_to_nodes, "scatter_point_attrs_to_nodes (CPU)");
  m.def("multiply_by_A", &multiply_by_A, "multiply by A (CPU)");
  m.def("multiply_by_AT", &multiply_by_AT, "multiply by AT (CPU)");
  m.def("multiply_by_G", &multiply_by_G, "multiply by AT (CPU)");
//   m.def("multiply_by_A_cpu_debug_single_thread", &multiply_by_A_cpu_debug_single_thread, "multiply by A (CPU)");
}

