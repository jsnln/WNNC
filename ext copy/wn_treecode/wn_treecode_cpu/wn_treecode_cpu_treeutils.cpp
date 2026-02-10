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


#include "wn_treecode_cpu.h"
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

    signedindex_t serial_index = -1;
    signedindex_t depth = -1;
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

    node->c_x = c_x;
    node->c_y = c_y;
    node->c_z = c_z;
    node->half_w = half_w;
    node->point_indices = point_indices;
    node->parent = parent;
    node->depth = cur_depth;

    cur_node_index += 1;

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

//////////// instantiation ////////////
auto ptr_build_tree_cpu_recursive_float  = build_tree_cpu_recursive<float>;
auto ptr_build_tree_cpu_recursive_double = build_tree_cpu_recursive<double>;
auto ptr_compute_tree_attributes_float  = compute_tree_attributes<float>;
auto ptr_compute_tree_attributes_double = compute_tree_attributes<double>;
auto ptr_serialize_tree_recursive_float  = serialize_tree_recursive<float>;
auto ptr_serialize_tree_recursive_double = serialize_tree_recursive<double>;
auto ptr_free_tree_recursive_float  = free_tree_recursive<float>;
auto ptr_free_tree_recursive_double = free_tree_recursive<double>;
