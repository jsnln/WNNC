import torch

class WindingNumberTreecode:
    def __init__(self,
                 points: torch.Tensor,
                 max_tree_depth=15):
        """
        points: [N, 3]
        """

        assert len(points.shape) == 2
        assert points.shape[1] == 3

        import wn_treecode._cpu  # necessary, because tree build is cpu either way
        self.is_cuda = points.is_cuda   # undefined behavior if changed after init
        if self.is_cuda:
            import wn_treecode._cuda # not necessary for cpu only

        self.treecode_package = wn_treecode._cuda if self.is_cuda else wn_treecode._cpu
        self.device = points.device

        tree_depth = max_tree_depth
        tree_packed = wn_treecode._cpu.build_tree(points.cpu(), tree_depth)   # tree build is on CPU either way

        if self.is_cuda:
            for i in range(len(tree_packed)):
                tree_packed[i] = tree_packed[i].to(self.device)
        node_parent_list, node_children_list, node_is_leaf_list, node_half_w_list, num_points_in_node, node2point_index, node2point_indexstart = tree_packed
        
        # if widths is not None:
        #     self.widths = widths.clone().to(self.device)
        # else:
        #     self.widths = torch.ones(points.shape[0], device=self.device).float() * 0.006    # a somewhat working value, certainly not optimal
        self.node_parent_list = node_parent_list
        self.node_children_list = node_children_list
        self.points = points
        self.node2point_index = node2point_index
        self.node2point_indexstart = node2point_indexstart
        self.num_points_in_node = num_points_in_node
        self.node_is_leaf_list = node_is_leaf_list
        self.node_half_w_list = node_half_w_list
        self.tree_depth = tree_depth

    def forward_A(self, normals, widths):
        """
        normals: [N, 3]
        widths: [N,]
        """
        assert self.points.shape == normals.shape
        assert len(widths.shape) == 1
        assert self.points.shape[0] == widths.shape[0]
        point_weights = (normals ** 2).sum(-1).sqrt()
        node_normals, node_reppoints, _ = \
            self.treecode_package.scatter_point_attrs_to_nodes(self.node_parent_list,
                                                    self.node_children_list,
                                                    self.points,
                                                    point_weights,
                                                    normals,
                                                    self.node2point_index,
                                                    self.node2point_indexstart,
                                                    self.num_points_in_node,
                                                    self.node_is_leaf_list,
                                                    self.tree_depth)
        out_vals = self.treecode_package.multiply_by_A(
            self.points,
            widths,
            self.points,
            normals,
            self.node2point_index,
            self.node2point_indexstart,
            self.node_children_list,
            node_normals,
            self.node_is_leaf_list,
            self.node_half_w_list,
            node_reppoints,
            self.num_points_in_node,
        )

        return out_vals
    
    def forward_AT(self, values, widths):
        """
        values: [N, 1]
        widths: [N,]
        """
        assert len(values.shape) == 2
        assert values.shape[0] == self.points.shape[0]
        assert values.shape[1] == 1
        assert len(widths.shape) == 1
        assert self.points.shape[0] == widths.shape[0]
        point_weights = (values ** 2).sum(-1).sqrt()
        node_scalars, node_reppoints, _  = \
            self.treecode_package.scatter_point_attrs_to_nodes(self.node_parent_list,
                                                    self.node_children_list,
                                                    self.points,
                                                    point_weights,
                                                    values,
                                                    self.node2point_index,
                                                    self.node2point_indexstart,
                                                    self.num_points_in_node,
                                                    self.node_is_leaf_list,
                                                    self.tree_depth)
        
        out_vecs = self.treecode_package.multiply_by_AT(
            self.points,
            widths,
            self.points,
            values,
            self.node2point_index,
            self.node2point_indexstart,
            self.node_children_list,
            node_scalars,
            self.node_is_leaf_list,
            self.node_half_w_list,
            node_reppoints,
            self.num_points_in_node,
        )

        return out_vecs
    
    def forward_G(self, normals, widths):
        """
        normals: [N, 3]
        widths: [N,]
        """
        assert self.points.shape == normals.shape
        assert len(widths.shape) == 1
        assert self.points.shape[0] == widths.shape[0]
        
        point_weights = (normals ** 2).sum(-1).sqrt()
        node_normals, node_reppoints, _ = \
            self.treecode_package.scatter_point_attrs_to_nodes(self.node_parent_list,
                                                    self.node_children_list,
                                                    self.points,
                                                    point_weights,
                                                    normals,
                                                    self.node2point_index,
                                                    self.node2point_indexstart,
                                                    self.num_points_in_node,
                                                    self.node_is_leaf_list,
                                                    self.tree_depth)
        out_normals = self.treecode_package.multiply_by_G(
            self.points,
            widths,
            self.points,
            normals,
            self.node2point_index,
            self.node2point_indexstart,
            self.node_children_list,
            node_normals,
            self.node_is_leaf_list,
            self.node_half_w_list,
            node_reppoints,
            self.num_points_in_node,
        )

        return out_normals