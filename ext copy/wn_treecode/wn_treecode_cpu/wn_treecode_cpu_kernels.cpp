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
#include <assert.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <omp.h>

// inplace elementwise addition to a consective block of memory defined by SPATIAL_DIM
// user is responsible for making sure dim == 3
template<typename scalar_t>
void add_vec_(scalar_t* dst, const scalar_t* src, signedindex_t n_dims) {
    for (signedindex_t d = 0; d < n_dims; d++) { dst[d] += src[d]; }
}
template<typename scalar_t>
void add_vec_(scalar_t* dst, const scalar_t* src, scalar_t scale, signedindex_t n_dims) {
    for (signedindex_t d = 0; d < n_dims; d++) { dst[d] += (src[d] * scale); }
}
// elementwise subtraction to a consective block of memory defined by SPATIAL_DIM
template<typename scalar_t>
void subtract_vec(scalar_t* dst, const scalar_t* src1, const scalar_t* src2, signedindex_t n_dims) {
    for (signedindex_t d = 0; d < n_dims; d++) { dst[d] = src1[d] - src2[d]; }
}
// elementwise assignment to a consective block of memory defined by SPATIAL_DIM
// user is responsible for making sure dim == 3
template<typename scalar_t>
void assign_vec(scalar_t* dst, const scalar_t* src, signedindex_t n_dims) {
    for (signedindex_t d = 0; d < n_dims; d++) { dst[d] = src[d]; }
}
template<typename scalar_t>
void assign_vec(scalar_t* dst, const scalar_t* src, scalar_t scale, signedindex_t n_dims) {
    for (signedindex_t d = 0; d < n_dims; d++) { dst[d] = (src[d] * scale); }
}

template<typename scalar_t>
scalar_t inner_prod(const scalar_t* vec1, const scalar_t* vec2, signedindex_t n_dims) {
    scalar_t result = 0;
    for (signedindex_t d = 0; d < n_dims; d++) { result += vec1[d] * vec2[d]; }
    return result;
}

template<typename scalar_t>
scalar_t get_point2point_dist2(const scalar_t* point1, const scalar_t* point2) {
    // squared distance between 2 points, both dim == 3
    // user is responsible for making sure dim == 3
    scalar_t dist2 = 0.0;
    for (signedindex_t d = 0; d < SPATIAL_DIM; d++) {
        dist2 += std::pow(point1[d] - point2[d], scalar_t(2.0));
    }
    return dist2;
}

template<typename scalar_t>
scalar_t eval_A_mu(const scalar_t* diff, const scalar_t* mu, scalar_t smooth_width, bool continuous_kernel=false) {
    // diff = x - y
    // returns -(x - y)\cdot mu
    // user is responsible for making sure dim == 3
    scalar_t dist = 0.0, dist2 = 0.0;    // d, d^2
    scalar_t denominator = 0.0;    // d^3 or w^3
    scalar_t result = 0.0;

    for (signedindex_t d = 0; d < SPATIAL_DIM; d++) {
        dist2 += (diff[d] * diff[d]); // d^2
    }
    dist = sqrt(dist2);  // d

    if (dist >= smooth_width) { // outside smoothing range
        denominator = dist * dist2;
        for (signedindex_t d = 0; d < SPATIAL_DIM; d++) {
            result += (-1 * diff[d] * mu[d]) / denominator;
        }
    } else if (continuous_kernel) {                    // inside smoothing range
        denominator = smooth_width * smooth_width * smooth_width;
            for (signedindex_t d = 0; d < SPATIAL_DIM; d++) {
            result += (-1 * diff[d] * mu[d]) / denominator;
        }
    }
    return result;
}



template<typename scalar_t>
void eval_AT_s_add_(scalar_t* out, const scalar_t* diff, const scalar_t* s, scalar_t smooth_width) {
    // diff = y - x
    // returns (y - x)*s (s is scalar)
    // user is responsible for making sure dim == 3
    scalar_t dist = 0.0, dist2 = 0.0;    // d, d^2
    scalar_t denominator = 0.0;    // d^3 or w^3
    // scalar_t result[SPATIAL_DIM] = 0.0;

    for (signedindex_t d = 0; d < SPATIAL_DIM; d++) {
        dist2 += (diff[d] * diff[d]); // d^2
    }
    dist = sqrt(dist2);  // d

    if (dist >= smooth_width) { // outside smoothing range
        denominator = dist * dist2;
        for (signedindex_t d = 0; d < SPATIAL_DIM; d++) {
            out[d] += (diff[d] * (*s)) / denominator;
        }
    } else {                    // inside smoothing range
        // denominator = smooth_width * smooth_width * smooth_width;
        // for (signedindex_t d = 0; d < SPATIAL_DIM; d++) {
        //     out[d] += (diff[d] * (*s)) / denominator;
        // }
    }
    return;
}



template<typename scalar_t>
void eval_G_mu_add_(scalar_t* out, const scalar_t* diff, const scalar_t* mu, scalar_t smooth_width) {
    // diff = x - y
    // returns -H(x-y)\cdot mu
    // user is responsible for making sure dim == 3
    scalar_t dist = 0.0, dist2 = 0.0, dist3 = 0.0, dist5 = 0.0;    // d, d^2

    // if (dist2 != 0.0) {
    //     printf("[DEBUG] dist2: %f\n", dist2);
    // }
    for (signedindex_t d = 0; d < SPATIAL_DIM; d++) {
        dist2 += (diff[d] * diff[d]); // d^2
    }
    dist = sqrt(dist2);  // d
    dist3 = dist * dist2;
    dist5 = dist2 * dist3;

    scalar_t diff_dot_mu = inner_prod<scalar_t>(diff, mu, SPATIAL_DIM);
    if (dist >= smooth_width) { // outside smoothing range
        for (signedindex_t d = 0; d < SPATIAL_DIM; d++) {
            out[d] += (mu[d] / dist3 - 3 * diff[d] * diff_dot_mu / dist5);
        }
    } else {    // inside smoothing range, WON'T work!!!
        // dist2 = smooth_width * smooth_width;
        // dist3 = dist2 * smooth_width;
        // dist5 = dist2 * dist3;
        // for (signedindex_t d = 0; d < SPATIAL_DIM; d++) {
        //     out[d] += (mu[d] / dist3 - 3 * diff[d] * diff_dot_mu / dist5);
        // }
    }
}



/// @brief collect point attributes to nodes

template<typename scalar_t>
void scatter_point_attrs_to_nodes_leaf_cpu_kernel(
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
        signedindex_t num_nodes,
        signedindex_t node_index) {
    
    if (node_index < num_nodes) {
        ptr_scattered_mask[node_index] = false;

        if (ptr_node_is_leaf_list[node_index]) {
            ptr_scattered_mask[node_index] = true;

            // representative points
            scalar_t reppoint[SPATIAL_DIM] = {};
            scalar_t reppoint_zero[SPATIAL_DIM] = {};

            scalar_t total_weight = 0;
            for (signedindex_t j = 0; j < ptr_num_points_in_node[node_index]; j++) {
                signedindex_t point_index = ptr_node2point_index[ptr_node2point_indexstart[node_index] + j];

                // the user is resposible for assuring point weights are all positive
                total_weight += ptr_point_weights[point_index];
                add_vec_<scalar_t>(ptr_out_node_attrs + node_index*attr_dim, ptr_point_attrs + point_index*attr_dim, attr_dim);
                add_vec_<scalar_t>(reppoint, ptr_points + point_index*SPATIAL_DIM, ptr_point_weights[point_index], SPATIAL_DIM);
                add_vec_<scalar_t>(reppoint_zero, ptr_points + point_index*SPATIAL_DIM, SPATIAL_DIM);
            }

            if (total_weight > 0) {
                assign_vec<scalar_t>(reppoint, reppoint, scalar_t(1) / total_weight, SPATIAL_DIM);
            } else {
                assign_vec<scalar_t>(reppoint, reppoint_zero, scalar_t(1) / scalar_t(ptr_num_points_in_node[node_index]), SPATIAL_DIM);
            }
            ptr_out_node_weights[node_index] = total_weight;
            assign_vec<scalar_t>(ptr_out_node_reppoints + node_index*SPATIAL_DIM, reppoint, SPATIAL_DIM);
        }
    }
}

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
        signedindex_t num_nodes) {

    omp_set_num_threads(20);
    #pragma omp parallel for
    for (signedindex_t node_index = 0; node_index < num_nodes; node_index++) {
        scatter_point_attrs_to_nodes_leaf_cpu_kernel<scalar_t>(
            ptr_node_parent_list,
            ptr_points,
            ptr_point_weights,
            ptr_point_attrs,
            ptr_node2point_index,
            ptr_node2point_indexstart,
            ptr_num_points_in_node,
            ptr_node_is_leaf_list,
            ptr_scattered_mask,

            ptr_out_node_attrs,
            ptr_out_node_reppoints,
            ptr_out_node_weights,

            attr_dim,
            num_nodes,
            node_index
        );
    }

}


/////////////////////////////////


template<typename scalar_t>
void scatter_point_attrs_to_nodes_nonleaf_cpu_kernel(
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
        signedindex_t num_nodes,
        signedindex_t node_index 
    ) {
    // signedindex_t node_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (node_index < num_nodes) {
        /// @note scatter only if it hasn't been scattered && it's not a leaf && all of its children have been scattered
        if (ptr_next_to_scatter_mask[node_index]) {

            ptr_scattered_mask[node_index] = true;

            scalar_t reppoint[SPATIAL_DIM] = {};
            scalar_t reppoint_zero[SPATIAL_DIM] = {};

            scalar_t total_weight = 0;
            for (signedindex_t k = 0; k < NUM_OCT_CHILDREN; k++) {
                signedindex_t child_index = ptr_node_children_list[node_index*NUM_OCT_CHILDREN + k];
                if (child_index != -1) {
                    
                    total_weight += ptr_out_node_weights[child_index];
                    add_vec_<scalar_t>(ptr_out_node_attrs + node_index*attr_dim, ptr_out_node_attrs + child_index*attr_dim, attr_dim);
                    add_vec_<scalar_t>(reppoint, ptr_out_node_reppoints + child_index*SPATIAL_DIM, ptr_out_node_weights[child_index], SPATIAL_DIM);
                    add_vec_<scalar_t>(reppoint_zero, ptr_out_node_reppoints + child_index*SPATIAL_DIM, SPATIAL_DIM);
                }
            }

            if (total_weight > 0) {
                assign_vec<scalar_t>(reppoint, reppoint, scalar_t(1) / total_weight, SPATIAL_DIM);
            } else {
                assign_vec<scalar_t>(reppoint, reppoint_zero, scalar_t(1) / scalar_t(ptr_num_points_in_node[node_index]), SPATIAL_DIM);
            }

            ptr_out_node_weights[node_index] = total_weight;
            assign_vec<scalar_t>(ptr_out_node_reppoints + node_index*SPATIAL_DIM, reppoint, SPATIAL_DIM);
        }
    }
}

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
        signedindex_t num_nodes) {
    omp_set_num_threads(20);
    #pragma omp parallel for
    for (signedindex_t node_index = 0; node_index < num_nodes; node_index++) {
        scatter_point_attrs_to_nodes_nonleaf_cpu_kernel<scalar_t>(
            ptr_node_parent_list,
            ptr_node_children_list,
            ptr_points,
            ptr_point_weights,
            ptr_point_attrs,
            ptr_node2point_index,
            ptr_node2point_indexstart,
            ptr_num_points_in_node,
            ptr_node_is_leaf_list,
            ptr_scattered_mask,
            ptr_next_to_scatter_mask,

            ptr_out_node_attrs,
            ptr_out_node_reppoints,
            ptr_out_node_weights,

            attr_dim,
            num_nodes,
            node_index); 
    }
}


////////////////

template<typename scalar_t>
void find_next_to_scatter_cpu_kernel(
        const signedindex_t* ptr_node_children_list,
        const bool* ptr_node_is_leaf_list,
        bool* ptr_scattered_mask,
        bool* ptr_next_to_scatter_mask,
        const signedindex_t* node2point_index,
        signedindex_t num_nodes,
        signedindex_t node_index
    ) {
    // signedindex_t node_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (node_index < num_nodes) {

        ptr_next_to_scatter_mask[node_index] = false;
        bool all_children_scattered = true;

        for (signedindex_t k = 0; k < NUM_OCT_CHILDREN; k++) {
            signedindex_t child_index = ptr_node_children_list[node_index*NUM_OCT_CHILDREN + k];
            if (child_index != -1) {
                if (!(ptr_scattered_mask[child_index])) {
                    all_children_scattered = false;
                }
            }
        }

        if ((!(ptr_scattered_mask[node_index])) && (!ptr_node_is_leaf_list[node_index]) && all_children_scattered) {
            ptr_next_to_scatter_mask[node_index] = true;
        }
    }
}


template<typename scalar_t>
void find_next_to_scatter_cpu_kernel_launcher(
        const signedindex_t* ptr_node_children_list,
        const bool* ptr_node_is_leaf_list,
        bool* ptr_scattered_mask,
        bool* ptr_next_to_scatter_mask,
        const signedindex_t* node2point_index,
        signedindex_t num_nodes
    ) {
    omp_set_num_threads(20);
    #pragma omp parallel for
    for (signedindex_t node_index = 0; node_index < num_nodes; node_index++) {
        find_next_to_scatter_cpu_kernel<scalar_t>(
            ptr_node_children_list,
            ptr_node_is_leaf_list,
            ptr_scattered_mask,
            ptr_next_to_scatter_mask,
            node2point_index,
            num_nodes,
            node_index
        );
    }
}


//////////////////////////////////

template<typename scalar_t>
void multiply_by_A_cpu_kernel(
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
        signedindex_t query_index,
        bool continuous_kernel=false
    ) {
    // the caller is responsible for making sure 'point_attrs' is [N, C=3]
    
    if (query_index < num_queries) {
        scalar_t out_val = 0.0;
        
        constexpr signedindex_t search_stack_max_size = ALLOWED_MAX_DEPTH*(NUM_OCT_CHILDREN - 1) + 1;
        signedindex_t search_stack[search_stack_max_size] = {};
        signedindex_t search_stack_top = 0;

        // a push
        assert(search_stack_top < search_stack_max_size);
        search_stack[search_stack_top++] = 0;
        while (search_stack_top > 0) {
            // a pop
            signedindex_t cur_node_index = search_stack[--search_stack_top];

            scalar_t point2node_dist2 = get_point2point_dist2(query_points + query_index*SPATIAL_DIM,
                                                              node_reppoints + cur_node_index*SPATIAL_DIM);

            /// @case 1: the query point is far from the sample, approximate the query value with the node center
            if (point2node_dist2 > std::pow(scalar_t(TREECODE_THRESHOLD * 2.0f) * node_half_w_list[cur_node_index], scalar_t(2.0f))) {
                scalar_t diff[SPATIAL_DIM];     // x - y
                subtract_vec<scalar_t>(diff, query_points + query_index*SPATIAL_DIM, node_reppoints + cur_node_index*SPATIAL_DIM, SPATIAL_DIM);
                out_val += eval_A_mu<scalar_t>(diff, node_attrs + cur_node_index * SPATIAL_DIM, query_width[query_index], continuous_kernel);
            } else {
                /// @case 2: the query point is not that far,
                //           if nonleaf, push children to the search stack
                if (!node_is_leaf_list[cur_node_index]) {
                    for (signedindex_t k = 0; k < NUM_OCT_CHILDREN; k++) {
                        if (node_children_list[cur_node_index * NUM_OCT_CHILDREN + k] != -1) {
                            assert(search_stack_top < search_stack_max_size);
                            search_stack[search_stack_top++] = node_children_list[cur_node_index * NUM_OCT_CHILDREN + k];
                        }
                    }
                } else {  /// @case 3: this node is a leaf node, compute over samples
                    for (signedindex_t k = 0; k < num_points_in_node[cur_node_index]; k++) {
                        signedindex_t point_index = node2point_index[node2point_indexstart[cur_node_index] + k];
                        scalar_t diff[SPATIAL_DIM];     // x - y
                        subtract_vec<scalar_t>(diff, query_points + query_index*SPATIAL_DIM, points + point_index*SPATIAL_DIM, SPATIAL_DIM);
                        out_val += eval_A_mu<scalar_t>(diff, point_attrs + point_index * SPATIAL_DIM, query_width[query_index], continuous_kernel);
                    }
                }
            }
        }
        out_attrs[query_index] = out_val;
    }
}


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
        bool continutous_kernel) {

    omp_set_num_threads(20);
    #pragma omp parallel for
    for (signedindex_t query_index = 0; query_index < num_queries; query_index++) {
        multiply_by_A_cpu_kernel<scalar_t>(
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
            num_points_in_node,
            out_attrs,           // [N,]
            num_queries,
            query_index,
            continutous_kernel);
    }
}


///////////////////

template<typename scalar_t>
void multiply_by_AT_cpu_kernel(
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
        signedindex_t num_queries,
        signedindex_t query_index
    ) {
    // the caller is responsible for making sure 'point_attrs' is [N, C=3]
    
    // signedindex_t query_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (query_index < num_queries) {
        scalar_t out_vec[SPATIAL_DIM] = {};
        
        constexpr signedindex_t search_stack_max_size = ALLOWED_MAX_DEPTH*(NUM_OCT_CHILDREN - 1) + 1;
        signedindex_t search_stack[search_stack_max_size] = {};
        signedindex_t search_stack_top = 0;

        // a push
        assert(search_stack_top < search_stack_max_size);
        search_stack[search_stack_top++] = 0;
        while (search_stack_top > 0) {
            // a pop
            signedindex_t cur_node_index = search_stack[--search_stack_top];
            scalar_t point2node_dist2 = get_point2point_dist2(query_points + query_index*SPATIAL_DIM,
                                                              node_reppoints + cur_node_index*SPATIAL_DIM);

            /// @case 1: the query point is far from the sample, approximate the query value with the node center
            if (point2node_dist2 > std::pow(scalar_t(TREECODE_THRESHOLD * 2.0f) * node_half_w_list[cur_node_index], scalar_t(2.0f))) {
                scalar_t diff[SPATIAL_DIM];     // x - y
                subtract_vec<scalar_t>(diff, query_points + query_index*SPATIAL_DIM, node_reppoints + cur_node_index*SPATIAL_DIM, SPATIAL_DIM);
                eval_AT_s_add_<scalar_t>(out_vec, diff, node_attrs + cur_node_index, query_width[query_index]);
                // printf("[DEBUG] got node contribution: %.4e from %d\n", *(node_attrs + cur_node_index), cur_node_index);
                // printf("        current_vec: (%f, %f, %f)\n", out_vec[0], out_vec[1], out_vec[2]);
                // printf("        diff: (%f, %f, %f), %f\n", diff[0], diff[1], diff[2], query_width[query_index]);
            } else {
                /// @case 2: the query point is not that far,
                //           if nonleaf, push children to the search stack
                if (!node_is_leaf_list[cur_node_index]) {
                    for (signedindex_t k = 0; k < NUM_OCT_CHILDREN; k++) {
                        if (node_children_list[cur_node_index * NUM_OCT_CHILDREN + k] != -1) {
                            assert(search_stack_top < search_stack_max_size);
                            search_stack[search_stack_top++] = node_children_list[cur_node_index * NUM_OCT_CHILDREN + k];
                        }
                    }
                } else {  /// @case 3: this node is a leaf node, compute over samples
                    for (signedindex_t k = 0; k < num_points_in_node[cur_node_index]; k++) {
                        signedindex_t point_index = node2point_index[node2point_indexstart[cur_node_index] + k];
                        scalar_t diff[SPATIAL_DIM];     // x - y
                        subtract_vec<scalar_t>(diff, query_points + query_index*SPATIAL_DIM, points + point_index*SPATIAL_DIM, SPATIAL_DIM);
                        eval_AT_s_add_<scalar_t>(out_vec, diff, point_attrs + point_index, query_width[query_index]);
                        // printf("[DEBUG] got point contribution: %.4e from %d\n", *(point_attrs + point_index), point_index);
                        // printf("        current_vec: (%f, %f, %f)\n", out_vec[0], out_vec[1], out_vec[2]);
                        // printf("        diff: (%f, %f, %f), %f\n", diff[0], diff[1], diff[2], query_width[query_index]);
                    }
                }
            }
        }
        assign_vec<scalar_t>(out_attrs + query_index*SPATIAL_DIM, out_vec, SPATIAL_DIM);
    }
}

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
        signedindex_t num_queries) {

    omp_set_num_threads(20);
    #pragma omp parallel for
    for (signedindex_t query_index = 0; query_index < num_queries; query_index++) {
        multiply_by_AT_cpu_kernel<scalar_t>(
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
            num_points_in_node,
            out_attrs,           // [N, 3]
            num_queries,
            query_index);
    }
}



/// @note getting negative gradient
template<typename scalar_t>
void multiply_by_G_cpu_kernel(
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
        signedindex_t num_queries,
        signedindex_t query_index
    ) {
    // the caller is responsible for making sure 'point_attrs' is [N, C=3]
    
    //  = blockDim.x * blockIdx.x + threadIdx.x;
    if (query_index < num_queries) {
    
    
        scalar_t out_vec[SPATIAL_DIM] = {};
        
        constexpr signedindex_t search_stack_max_size = ALLOWED_MAX_DEPTH*(NUM_OCT_CHILDREN - 1) + 1;
        signedindex_t search_stack[search_stack_max_size] = {};
        signedindex_t search_stack_top = 0;

        // a push
        assert(search_stack_top < search_stack_max_size);
        search_stack[search_stack_top++] = 0;
        while (search_stack_top > 0) {
            // a pop
            signedindex_t cur_node_index = search_stack[--search_stack_top];
            scalar_t point2node_dist2 = get_point2point_dist2(query_points + query_index*SPATIAL_DIM,
                                                              node_reppoints + cur_node_index*SPATIAL_DIM);

            /// @case 1: the query point is far from the sample, approximate the query value with the node center
            if (point2node_dist2 > std::pow(scalar_t(TREECODE_THRESHOLD * 2.0f) * node_half_w_list[cur_node_index], scalar_t(2.0f))) {
                scalar_t diff[SPATIAL_DIM];     // x - y
                subtract_vec<scalar_t>(diff, query_points + query_index*SPATIAL_DIM, node_reppoints + cur_node_index*SPATIAL_DIM, SPATIAL_DIM);
                eval_G_mu_add_<scalar_t>(out_vec, diff, node_attrs + cur_node_index*SPATIAL_DIM, query_width[query_index]);
            } else {
                /// @case 2: the query point is not that far,
                //           if nonleaf, push children to the search stack
                if (!node_is_leaf_list[cur_node_index]) {
                    for (signedindex_t k = 0; k < NUM_OCT_CHILDREN; k++) {
                        if (node_children_list[cur_node_index * NUM_OCT_CHILDREN + k] != -1) {
                            assert(search_stack_top < search_stack_max_size);
                            search_stack[search_stack_top++] = node_children_list[cur_node_index * NUM_OCT_CHILDREN + k];
                        }
                    }
                } else {  /// @case 3: this node is a leaf node, compute over samples
                    for (signedindex_t k = 0; k < num_points_in_node[cur_node_index]; k++) {
                        signedindex_t point_index = node2point_index[node2point_indexstart[cur_node_index] + k];
                        scalar_t diff[SPATIAL_DIM];     // x - y
                        subtract_vec<scalar_t>(diff, query_points + query_index*SPATIAL_DIM, points + point_index*SPATIAL_DIM, SPATIAL_DIM);
                        eval_G_mu_add_<scalar_t>(out_vec, diff, point_attrs + point_index*SPATIAL_DIM, query_width[query_index]);
                    }
                }
            }
        }
        assign_vec<scalar_t>(out_attrs + query_index*SPATIAL_DIM, out_vec, SPATIAL_DIM);
    }
}

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
        signedindex_t num_queries) {
    omp_set_num_threads(20);
    #pragma omp parallel for
    for (signedindex_t query_index = 0; query_index < num_queries; query_index++) {
        multiply_by_G_cpu_kernel<scalar_t>(
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
            num_points_in_node,
            out_attrs,           // [N, 3]
            num_queries,
            query_index);
    }
}


//////////// instantiation ////////////
auto ptr_scatter_point_attrs_to_nodes_leaf_cpu_kernel_launcher_float  = scatter_point_attrs_to_nodes_leaf_cpu_kernel_launcher<float>;
auto ptr_scatter_point_attrs_to_nodes_leaf_cpu_kernel_launcher_double = scatter_point_attrs_to_nodes_leaf_cpu_kernel_launcher<double>;
auto ptr_scatter_point_attrs_to_nodes_nonleaf_cpu_kernel_launcher_float  = scatter_point_attrs_to_nodes_nonleaf_cpu_kernel_launcher<float>;
auto ptr_scatter_point_attrs_to_nodes_nonleaf_cpu_kernel_launcher_double = scatter_point_attrs_to_nodes_nonleaf_cpu_kernel_launcher<double>;
auto ptr_find_next_to_scatter_cpu_kernel_launcher_float  = find_next_to_scatter_cpu_kernel_launcher<float>;
auto ptr_find_next_to_scatter_cpu_kernel_launcher_double = find_next_to_scatter_cpu_kernel_launcher<double>;
auto ptr_multiply_by_A_cpu_kernel_launcher_float  = multiply_by_A_cpu_kernel_launcher<float>;
auto ptr_multiply_by_A_cpu_kernel_launcher_double = multiply_by_A_cpu_kernel_launcher<double>;
auto ptr_multiply_by_AT_cpu_kernel_launcher_float  = multiply_by_AT_cpu_kernel_launcher<float>;
auto ptr_multiply_by_AT_cpu_kernel_launcher_double = multiply_by_AT_cpu_kernel_launcher<double>;
auto ptr_multiply_by_G_cpu_kernel_launcher_float  = multiply_by_G_cpu_kernel_launcher<float>;
auto ptr_multiply_by_G_cpu_kernel_launcher_double = multiply_by_G_cpu_kernel_launcher<double>;