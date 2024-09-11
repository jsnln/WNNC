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

#include "Octree.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <string>
#include <fstream>
#include "Geometry.h"
#include <CLI11.hpp>
// #include <cnpy.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

#include "wn_treecode_cpu.h"
#include "wn_treecode_cuda.h"

typedef float used_dtype;

struct SerializedTree {

	size_t num_nodes = 0;
	signedindex_t tree_depth = -1;

	signedindex_t * const node_parent_list_ptr; // = std::vector<signedindex_t>(num_nodes, 0);
	signedindex_t * const node_children_list_ptr; // = std::vector<signedindex_t>(num_nodes * NUM_OCT_CHILDREN, 0);
	bool * const node_is_leaf_list_ptr; // = new bool[num_nodes];

	signedindex_t * const num_points_in_node_ptr; // = std::vector<signedindex_t>(num_nodes, 0);
    signedindex_t * const node2point_indexstart_ptr; // = std::vector<signedindex_t>(num_nodes, 0);
    used_dtype * const node_half_w_list_ptr; // = std::vector<used_dtype>(num_nodes, 0);

	SerializedTree(size_t in_num_nodes, signedindex_t in_tree_depth):
		num_nodes(in_num_nodes),
		tree_depth(in_tree_depth),
		node_parent_list_ptr(new signedindex_t[num_nodes]),
		node_children_list_ptr(new signedindex_t[num_nodes * NUM_OCT_CHILDREN]),
		node_is_leaf_list_ptr(new bool[num_nodes]),
		num_points_in_node_ptr(new signedindex_t[num_nodes]),
		node2point_indexstart_ptr(new signedindex_t[num_nodes]),
		node_half_w_list_ptr(new used_dtype[num_nodes])
	{
		std::memset(node_parent_list_ptr, 		0, num_nodes * sizeof(signedindex_t));
		std::memset(node_children_list_ptr, 	0, num_nodes * NUM_OCT_CHILDREN * sizeof(signedindex_t));
		std::memset(node_is_leaf_list_ptr, 		0, num_nodes * sizeof(bool));
		std::memset(num_points_in_node_ptr, 	0, num_nodes * sizeof(signedindex_t));
		std::memset(node2point_indexstart_ptr, 	0, num_nodes * sizeof(signedindex_t));
		std::memset(node_half_w_list_ptr, 		0, num_nodes * sizeof(used_dtype));
	}
	SerializedTree(const SerializedTree & other):
		num_nodes(other.num_nodes),
		tree_depth(other.tree_depth),
		node_parent_list_ptr(new signedindex_t[num_nodes]),
		node_children_list_ptr(new signedindex_t[num_nodes * NUM_OCT_CHILDREN]),
		node_is_leaf_list_ptr(new bool[num_nodes]),
		num_points_in_node_ptr(new signedindex_t[num_nodes]),
		node2point_indexstart_ptr(new signedindex_t[num_nodes]),
		node_half_w_list_ptr(new used_dtype[num_nodes])
	{
		std::memcpy(node_parent_list_ptr, 		other.node_parent_list_ptr, 		num_nodes * sizeof(signedindex_t));
		std::memcpy(node_children_list_ptr, 	other.node_children_list_ptr, 		num_nodes * NUM_OCT_CHILDREN * sizeof(signedindex_t));
		std::memcpy(node_is_leaf_list_ptr, 		other.node_is_leaf_list_ptr, 		num_nodes * sizeof(bool));
		std::memcpy(num_points_in_node_ptr, 	other.num_points_in_node_ptr, 		num_nodes * sizeof(signedindex_t));
		std::memcpy(node2point_indexstart_ptr, 	other.node2point_indexstart_ptr, 	num_nodes * sizeof(signedindex_t));
		std::memcpy(node_half_w_list_ptr, 		other.node_half_w_list_ptr, 		num_nodes * sizeof(used_dtype));
	}
	~SerializedTree() {
		delete [] node_parent_list_ptr;
		delete [] node_children_list_ptr;
		delete [] node_is_leaf_list_ptr;
		delete [] num_points_in_node_ptr;
		delete [] node2point_indexstart_ptr;
		delete [] node_half_w_list_ptr;
	}
	// SerializedTree & operator=(const SerializedTree & other) {
		
	// }
};


struct SerializedTreeCUDA {
	// same as above but all pointers point to CUDA memory

	size_t num_nodes = 0;
	signedindex_t tree_depth = -1;

	signedindex_t * node_parent_list_ptr_d; // = std::vector<signedindex_t>(num_nodes, 0);
	signedindex_t * node_children_list_ptr_d; // = std::vector<signedindex_t>(num_nodes * NUM_OCT_CHILDREN, 0);
	bool * node_is_leaf_list_ptr_d; // = new bool[num_nodes];

	signedindex_t * num_points_in_node_ptr_d; // = std::vector<signedindex_t>(num_nodes, 0);
    signedindex_t * node2point_indexstart_ptr_d; // = std::vector<signedindex_t>(num_nodes, 0);
    used_dtype * node_half_w_list_ptr_d; // = std::vector<used_dtype>(num_nodes, 0);

    signedindex_t * node2point_index_ptr_d; // = std::vector<signedindex_t>(num_nodes, 0);

	// SerializedTree(size_t in_num_nodes, signedindex_t in_tree_depth):
	// 	num_nodes(in_num_nodes),
	// 	tree_depth(in_tree_depth),
	// 	node_parent_list_ptr(new signedindex_t[num_nodes]),
	// 	node_children_list_ptr(new signedindex_t[num_nodes * NUM_OCT_CHILDREN]),
	// 	node_is_leaf_list_ptr(new bool[num_nodes]),
	// 	num_points_in_node_ptr(new signedindex_t[num_nodes]),
	// 	node2point_indexstart_ptr(new signedindex_t[num_nodes]),
	// 	node_half_w_list_ptr(new used_dtype[num_nodes])
	// {
	// 	std::memset(node_parent_list_ptr, 		0, num_nodes * sizeof(signedindex_t));
	// 	std::memset(node_children_list_ptr, 	0, num_nodes * NUM_OCT_CHILDREN * sizeof(signedindex_t));
	// 	std::memset(node_is_leaf_list_ptr, 		0, num_nodes * sizeof(bool));
	// 	std::memset(num_points_in_node_ptr, 	0, num_nodes * sizeof(signedindex_t));
	// 	std::memset(node2point_indexstart_ptr, 	0, num_nodes * sizeof(signedindex_t));
	// 	std::memset(node_half_w_list_ptr, 		0, num_nodes * sizeof(used_dtype));
	// }
	SerializedTreeCUDA(const SerializedTree & other, const std::vector<signedindex_t> node2point_index):
		num_nodes(other.num_nodes),
		tree_depth(other.tree_depth) {
		
		cudaMalloc(&node_parent_list_ptr_d, 		num_nodes * sizeof(signedindex_t));
		cudaMalloc(&node_children_list_ptr_d, 		num_nodes * NUM_OCT_CHILDREN * sizeof(signedindex_t));
		cudaMalloc(&node_is_leaf_list_ptr_d, 		num_nodes * sizeof(bool));
		cudaMalloc(&num_points_in_node_ptr_d, 		num_nodes * sizeof(signedindex_t));
		cudaMalloc(&node2point_indexstart_ptr_d, 	num_nodes * sizeof(signedindex_t));
		cudaMalloc(&node_half_w_list_ptr_d, 		num_nodes * sizeof(used_dtype));
		cudaMalloc(&node2point_index_ptr_d, 		node2point_index.size() * sizeof(signedindex_t));

		cudaMemcpy(node_parent_list_ptr_d, 			other.node_parent_list_ptr, 		num_nodes * sizeof(signedindex_t),	cudaMemcpyHostToDevice);
		cudaMemcpy(node_children_list_ptr_d, 		other.node_children_list_ptr, 		num_nodes * NUM_OCT_CHILDREN * sizeof(signedindex_t),	cudaMemcpyHostToDevice);
		cudaMemcpy(node_is_leaf_list_ptr_d, 		other.node_is_leaf_list_ptr, 		num_nodes * sizeof(bool), 			cudaMemcpyHostToDevice);
		cudaMemcpy(num_points_in_node_ptr_d, 		other.num_points_in_node_ptr, 		num_nodes * sizeof(signedindex_t),	cudaMemcpyHostToDevice);
		cudaMemcpy(node2point_indexstart_ptr_d, 	other.node2point_indexstart_ptr, 	num_nodes * sizeof(signedindex_t), 	cudaMemcpyHostToDevice);
		cudaMemcpy(node_half_w_list_ptr_d, 			other.node_half_w_list_ptr, 		num_nodes * sizeof(used_dtype),		cudaMemcpyHostToDevice);		
		cudaMemcpy(node2point_index_ptr_d, 			node2point_index.data(),	 		node2point_index.size() * sizeof(signedindex_t), 	cudaMemcpyHostToDevice);
	}
	~SerializedTreeCUDA() {
		cudaFree(node_parent_list_ptr_d);
		cudaFree(node_children_list_ptr_d);
		cudaFree(node_is_leaf_list_ptr_d);
		cudaFree(num_points_in_node_ptr_d);
		cudaFree(node2point_indexstart_ptr_d);
		cudaFree(node_half_w_list_ptr_d);
		cudaFree(node2point_index_ptr_d);
	}
	// SerializedTree & operator=(const SerializedTree & other) {
		
	// }
};


/// @note @todo maybe Eigen is better, but I don't want to bother with it now.
std::pair<SerializedTree,				// the tree
		   std::vector<signedindex_t>>	// node2point index
build_tree(const std::vector<used_dtype> points_normalized, signedindex_t max_depth) {

    const auto num_points = points_normalized.size() / 3;
	cout << "[DEBUG] num_points: " << num_points << "\n";
    std::vector<signedindex_t> point_indices(num_points);
    for (signedindex_t i = 0; i < num_points; i++) {
        point_indices[i] = i;
    }

    signedindex_t cur_node_index = 0;
    auto root = build_tree_cpu_recursive<used_dtype>(
        points_normalized.data(),
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
    compute_tree_attributes<used_dtype>(root, num_nodes, num_leaves, tree_depth);
    std::cout << "num_nodes: " << num_nodes << ", num_leaves: " << num_leaves << ", tree depth: " << tree_depth << "\n";

    auto stdvec_node2point_index = std::vector<signedindex_t>();
	SerializedTree serialized_tree(num_nodes, tree_depth);

    serialize_tree_recursive(root,
							 serialized_tree.node_parent_list_ptr,
                             serialized_tree.node_children_list_ptr,
                             serialized_tree.node_is_leaf_list_ptr,
                             serialized_tree.node_half_w_list_ptr,
                             serialized_tree.num_points_in_node_ptr,
                             serialized_tree.node2point_indexstart_ptr,
                             stdvec_node2point_index);

    // auto node2point_index = torch::zeros({stdvec_node2point_index.size()}, long_tensor_options);
    // std::memcpy(node2point_index.data<signedindex_t>(), stdvec_node2point_index.data(), stdvec_node2point_index.size()*sizeof(signedindex_t));

    std::cout << "tree_depth*num_points: " << tree_depth*num_points << ", stdvec_node2point_index.size(): " << stdvec_node2point_index.size() << "\n";

    free_tree_recursive(root);

    return {serialized_tree, stdvec_node2point_index};
}

int main(int argc, char** argv) {

    std::string ply_suffix(".ply");
	std::string normalized_npy_suffix = "_normalized.npy";
	std::string query_npy_suffix("_for_query.npy");
    
	std::string inFileName;
	std::string outFileName;
	int minDepth = 1;
	int maxDepth = 10;

	used_dtype width = 0.01f;
    
    CLI::App app("PGRExportQuery");
    app.add_option("-i", inFileName, "input filename of xyz format")->required();
	app.add_option("-o", outFileName, "output filename with no suffix")->required();
	app.add_option("-w", width, "");
	app.add_option("-m", minDepth, "");
	app.add_option("-d", maxDepth, "");
	
    CLI11_PARSE(app, argc, argv);

	if (maxDepth < minDepth) {
		cout << "[In PGRExportQuery] WARNING: minDepth "
			 << minDepth
			 << " smaller than maxDepth "
			 << maxDepth
			 << ", ignoring given minDepth\n";
	}
		
	Octree tree;
	tree.setTree(inFileName, maxDepth, minDepth);//1382_seahorse2_p

	//*** Nodes for query are from gridDataVector *** START ***
	unsigned long N_query_pts = tree.gridDataVector.size();
	unsigned long N_sample_pts = tree.samplePoints.size();
	cout << "[DEBUG] samplePoints.size(): " << tree.samplePoints.size() << "\n";

	cout << "[DEBUG] N_sample_pts: " << N_sample_pts << "\n";

	// getting normalized point samples (PGR convention [0,1]^3 => WNNC convention [-1,1]^3)
	std::vector<used_dtype> wn_pts_input(N_sample_pts * 3);
	std::vector<used_dtype> wn_nml_input(N_sample_pts * 3);	// this should be the area-weighted normal
	std::vector<used_dtype> wn_widths_input(N_sample_pts * 3);	// for isovalue
	std::vector<used_dtype> wn_pts_weights(N_sample_pts);	// for scatter to node, = (normals ** 2).sum(-1).sqrt() == area
	
	for (int j = 0; j < N_sample_pts; j++) {
		wn_pts_input[3 * j + 0] = 2 * tree.samplePoints[j].x - 1;
		wn_pts_input[3 * j + 1] = 2 * tree.samplePoints[j].y - 1;
		wn_pts_input[3 * j + 2] = 2 * tree.samplePoints[j].z - 1;

		used_dtype nx, ny, nz, nlen, area;
		nx = tree.samplePoints[j].nx;
		ny = tree.samplePoints[j].ny;
		nz = tree.samplePoints[j].nz;
		nlen = std::max(std::sqrt(nx * nx + ny * ny + nz * nz), 1e-12f);
		// area = 4 * tree.samplePoints[j].area;
		area = 1e-5f;

		wn_nml_input[3 * j + 0] = ( nx / nlen * area );
		wn_nml_input[3 * j + 1] = ( ny / nlen * area );
		wn_nml_input[3 * j + 2] = ( nz / nlen * area );

		wn_pts_weights[j] = ( area );
		wn_widths_input[j] = ( width );	// using a fixed value here, per-point width is supported but the user needs to define it
	}

	// getting normalized point samples (PGR convention [0,1]^3 => WNNC convention [-1,1]^3)
	std::vector<used_dtype> wn_pts_query(N_query_pts * 3);
	std::vector<used_dtype> wn_widths_query(N_query_pts);
	for(int i=0; i < N_query_pts; i++) {
		wn_pts_query[3 * i + 0] = ( 2 * tree.gridDataVector[i]->coords[0] - 1 );
		wn_pts_query[3 * i + 1] = ( 2 * tree.gridDataVector[i]->coords[1] - 1 );
		wn_pts_query[3 * i + 2] = ( 2 * tree.gridDataVector[i]->coords[2] - 1 );

		wn_widths_query[i] = ( width );	// using a fixed value here, per-point width is supported but the user needs to define it
	}

	// octree for treecode winding number
	// C++17 structured binding:
	cout << "[DEBUG] wn_pts_input.size(): " << wn_pts_input.size() << "\n";
    const auto serialized_tree_and_node2point_index = build_tree(wn_pts_input, /* max_depth = */15);

	SerializedTreeCUDA serialized_tree_cuda(serialized_tree_and_node2point_index.first, serialized_tree_and_node2point_index.second);

    signedindex_t num_nodes = serialized_tree_and_node2point_index.first.num_nodes;
    signedindex_t attr_dim = SPATIAL_DIM;	// normal dim
    assert(attr_dim == SPATIAL_DIM or attr_dim == 1);

	/// why am I using new? because we have no std::vector<bool>
	bool * scattered_mask_ptr_d; // = new bool[num_nodes];
    bool * next_to_scatter_mask_ptr_d; // = new bool[num_nodes];
    used_dtype * out_node_attrs_ptr_d; // = new used_dtype[num_nodes * attr_dim];
    used_dtype * out_node_reppoints_ptr_d; // = new used_dtype[num_nodes * SPATIAL_DIM];
    used_dtype * out_node_weights_ptr_d; // = new used_dtype[num_nodes];
    used_dtype * wn_pts_input_ptr_d; // for wn_pts_input
    used_dtype * wn_pts_weights_ptr_d; // for wn_pts_weights
    used_dtype * wn_nml_input_ptr_d; // for 
    used_dtype * wn_widths_input_ptr_d; // for wn_pts_input
    used_dtype * wn_pts_query_ptr_d; // for wn_pts_input
    used_dtype * wn_widths_query_ptr_d; // for 
    
	used_dtype * wn_queried_ptr_d; // for 
    used_dtype * wn_queried_input_ptr_d; // for 

	cudaMalloc(&scattered_mask_ptr_d, 			num_nodes * sizeof(bool));
	cudaMalloc(&next_to_scatter_mask_ptr_d, 	num_nodes * sizeof(bool));
	cudaMalloc(&out_node_attrs_ptr_d, 			num_nodes * attr_dim * sizeof(used_dtype));
	cudaMalloc(&out_node_reppoints_ptr_d, 		num_nodes * SPATIAL_DIM * sizeof(used_dtype));
	cudaMalloc(&out_node_weights_ptr_d, 		num_nodes * sizeof(used_dtype));
	cudaMalloc(&wn_pts_input_ptr_d, 			N_sample_pts * SPATIAL_DIM * sizeof(used_dtype));
	cudaMalloc(&wn_pts_weights_ptr_d, 			N_sample_pts * sizeof(used_dtype));
	cudaMalloc(&wn_nml_input_ptr_d, 			N_sample_pts * attr_dim * sizeof(used_dtype));
	cudaMalloc(&wn_widths_input_ptr_d, 			N_sample_pts * sizeof(used_dtype));
	cudaMalloc(&wn_pts_query_ptr_d, 			N_query_pts * SPATIAL_DIM * sizeof(used_dtype));
	cudaMalloc(&wn_widths_query_ptr_d, 			N_query_pts * sizeof(used_dtype));
	
	cudaMalloc(&wn_queried_ptr_d, 				N_query_pts * sizeof(used_dtype));
	cudaMalloc(&wn_queried_input_ptr_d,			N_sample_pts * sizeof(used_dtype));

	cudaMemset(scattered_mask_ptr_d, 		0, 	num_nodes * sizeof(bool));
	cudaMemset(next_to_scatter_mask_ptr_d, 	0, 	num_nodes * sizeof(bool));
	cudaMemset(out_node_attrs_ptr_d, 		0, 	num_nodes * attr_dim * sizeof(used_dtype));
	cudaMemset(out_node_reppoints_ptr_d, 	0, 	num_nodes * SPATIAL_DIM * sizeof(used_dtype));
	cudaMemset(out_node_weights_ptr_d, 		0, 	num_nodes * sizeof(used_dtype));
	cudaMemcpy(wn_pts_input_ptr_d,		wn_pts_input.data(),	N_sample_pts * SPATIAL_DIM * sizeof(used_dtype), 	cudaMemcpyHostToDevice);
	cudaMemcpy(wn_pts_weights_ptr_d,	wn_pts_weights.data(),	N_sample_pts * sizeof(used_dtype),					cudaMemcpyHostToDevice);
	cudaMemcpy(wn_nml_input_ptr_d,		wn_nml_input.data(),	N_sample_pts * attr_dim * sizeof(used_dtype),		cudaMemcpyHostToDevice);
	cudaMemcpy(wn_widths_input_ptr_d,	wn_widths_input.data(),	N_sample_pts * sizeof(used_dtype),					cudaMemcpyHostToDevice);
	cudaMemcpy(wn_pts_query_ptr_d, 		wn_pts_query.data(),	N_query_pts * SPATIAL_DIM * sizeof(used_dtype),		cudaMemcpyHostToDevice);
	cudaMemcpy(wn_widths_query_ptr_d,	wn_widths_query.data(),	N_query_pts * sizeof(used_dtype),					cudaMemcpyHostToDevice);

	cudaMemset(wn_queried_ptr_d, 			0, 	N_query_pts * sizeof(used_dtype));
	cudaMemset(wn_queried_input_ptr_d, 		0, 	N_sample_pts * sizeof(used_dtype));

	// scatter_point_attrs_to_nodes_leaf_cpu_kernel_launcher<used_dtype>(
	// 	serialized_tree.node_parent_list_ptr,
	// 	wn_pts_input.data(),
	// 	wn_pts_weights.data(),
	// 	wn_nml_input.data(),
	// 	node2point_index.data(),
	// 	serialized_tree.node2point_indexstart_ptr,
	// 	serialized_tree.num_points_in_node_ptr,
	// 	serialized_tree.node_is_leaf_list_ptr,
	// 	scattered_mask_ptr,
	// 	out_node_attrs_ptr,
	// 	out_node_reppoints_ptr,
	// 	out_node_weights_ptr,
	// 	attr_dim,
	// 	num_nodes
	// );
	signedindex_t num_blocks = (num_nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	scatter_point_attrs_to_nodes_leaf_cuda_kernel<used_dtype><<<num_blocks, THREADS_PER_BLOCK>>>(
		serialized_tree_cuda.node_parent_list_ptr_d,
		wn_pts_input_ptr_d,
		wn_pts_weights_ptr_d,
		wn_nml_input_ptr_d,
		serialized_tree_cuda.node2point_index_ptr_d,
		serialized_tree_cuda.node2point_indexstart_ptr_d,
		serialized_tree_cuda.num_points_in_node_ptr_d,
		serialized_tree_cuda.node_is_leaf_list_ptr_d,
		scattered_mask_ptr_d,
		out_node_attrs_ptr_d,
		out_node_reppoints_ptr_d,
		out_node_weights_ptr_d,
		attr_dim,
		num_nodes
	);

    for (signedindex_t depth = serialized_tree_cuda.tree_depth-1; depth >= 0; depth--) {
		find_next_to_scatter_cuda_kernel<used_dtype><<<num_blocks, THREADS_PER_BLOCK>>>(
			serialized_tree_cuda.node_children_list_ptr_d,
			serialized_tree_cuda.node_is_leaf_list_ptr_d,
			scattered_mask_ptr_d,
			next_to_scatter_mask_ptr_d,
			serialized_tree_cuda.node2point_index_ptr_d,
			num_nodes
		);

		scatter_point_attrs_to_nodes_nonleaf_cuda_kernel<used_dtype><<<num_blocks, THREADS_PER_BLOCK>>>(
			serialized_tree_cuda.node_parent_list_ptr_d,
			serialized_tree_cuda.node_children_list_ptr_d,
			wn_pts_input_ptr_d,
			wn_pts_weights_ptr_d,
			wn_nml_input_ptr_d,
			serialized_tree_cuda.node2point_index_ptr_d,
			serialized_tree_cuda.node2point_indexstart_ptr_d,
			serialized_tree_cuda.num_points_in_node_ptr_d,
			serialized_tree_cuda.node_is_leaf_list_ptr_d,
			scattered_mask_ptr_d,
			next_to_scatter_mask_ptr_d,
			out_node_attrs_ptr_d,
			out_node_reppoints_ptr_d,
			out_node_weights_ptr_d,
			attr_dim,
			num_nodes
		);
    }

	// kernel launch
	std::vector<used_dtype> wn_queried(N_query_pts);
    num_blocks = (N_query_pts + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	multiply_by_A_cuda_kernel<used_dtype><<<num_blocks, THREADS_PER_BLOCK>>>(
		wn_pts_query_ptr_d,  // [N', 3]
		wn_widths_query_ptr_d,   // [N',]
		wn_pts_input_ptr_d,        // [N, 3]
		wn_nml_input_ptr_d,   // [N, C]
		serialized_tree_cuda.node2point_index_ptr_d,
		serialized_tree_cuda.node2point_indexstart_ptr_d,
		serialized_tree_cuda.node_children_list_ptr_d,
		out_node_attrs_ptr_d,
		serialized_tree_cuda.node_is_leaf_list_ptr_d,
		serialized_tree_cuda.node_half_w_list_ptr_d,
		out_node_reppoints_ptr_d,
		serialized_tree_cuda.num_points_in_node_ptr_d,
		wn_queried_ptr_d,           // [N, 3]
		N_query_pts,
		true
	);
	cudaMemcpy(wn_queried.data(), wn_queried_ptr_d, N_query_pts * sizeof(used_dtype), cudaMemcpyDeviceToHost);

	// for isovalue
	std::vector<used_dtype> wn_queried_at_input(N_sample_pts);
	num_blocks = (N_sample_pts + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	multiply_by_A_cuda_kernel<used_dtype><<<num_blocks, THREADS_PER_BLOCK>>>(
		wn_pts_input_ptr_d,  // [N', 3]
		wn_widths_input_ptr_d,   // [N',]
		wn_pts_input_ptr_d,        // [N, 3]
		wn_nml_input_ptr_d,   // [N, C]
		serialized_tree_cuda.node2point_index_ptr_d,
		serialized_tree_cuda.node2point_indexstart_ptr_d,
		serialized_tree_cuda.node_children_list_ptr_d,
		out_node_attrs_ptr_d,
		serialized_tree_cuda.node_is_leaf_list_ptr_d,
		serialized_tree_cuda.node_half_w_list_ptr_d,
		out_node_reppoints_ptr_d,
		serialized_tree_cuda.num_points_in_node_ptr_d,
		wn_queried_input_ptr_d,           // [N, 3]
		N_sample_pts,
		true
	);
	cudaMemcpy(wn_queried_at_input.data(), wn_queried_input_ptr_d, N_sample_pts * sizeof(used_dtype), cudaMemcpyDeviceToHost);

	// cnpy::npy_save(outFileName + normalized_npy_suffix, &pts_normalized[0], {N_sample_pts, 3}, "w");
	// std::cout << "[In PGRExportQuery] Normalizing the point cloud. Result saved to " << outFileName + normalized_npy_suffix <<std::endl;
	// cnpy::npy_save(outFileName + query_npy_suffix, &grid_coords[0], {N_grid_pts, 3}, "w");
	// std::cout << "[In PGRExportQuery] Exporting points on octree for query. Result saved to " << outFileName + query_npy_suffix <<std::endl;

	int N_grid = tree.gridDataVector.size();
	std::cout << "[DEBUG] N_grid: " << N_grid << std::endl;
	
	for(int idx=0; idx<N_grid; idx++) {
		tree.gridDataVector[idx]->value = -wn_queried[idx];
		tree.gridDataVector[idx]->smoothWidth = wn_widths_query[idx] / 2;
	}

	// tree.loadImplicitFunctionFromNPY(inGridValFileName, N_grid);
	// tree.loadGridWidthFromNPY(inGridWidthFileName, N_grid);

	std::nth_element(wn_queried_at_input.begin(), wn_queried_at_input.begin() + wn_queried_at_input.size() / 2, wn_queried_at_input.end());
	used_dtype isoValue = -wn_queried_at_input[wn_queried_at_input.size() / 2];

	std::cout << "[DEBUG] Isovalue: " << isoValue << std::endl;

	CoredVectorMeshData mesh;
	tree.initLeaf();
	std::cout << "[DEBUG] initLeaf done: " << std::endl;
	std::cout << "[DEBUG] num leaves: " << tree.root.leaves() << std::endl;
	tree.GetMCIsoTriangles(isoValue,  &mesh, 0, 1, 0, 0);
	std::cout << "[DEBUG] triangles got: " << isoValue << std::endl;
	char fileChar[255];
	strcpy(fileChar, (outFileName).c_str());
	tree.writePolygon2(&mesh, fileChar);
	std::cout << "[DEBUG] Polygon Written to " << outFileName << std::endl;
}
