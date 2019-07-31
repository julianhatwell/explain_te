# stats from forest_walker object
    def forest_stats_by_label(self, label = None):
        if label is None:
            idx = Series([True] * self.n_instances) # it's easier if has the same type as the labels
            label = 'all_classes'
        else:
            idx = self.labels == label
        idx = idx.values

        n_instances_lab = sum(idx) # number of instances having the current label
        if n_instances_lab == 0: return

        # object to hold all the statistics
        statistics = {}
        statistics['n_trees'] = self.n_trees
        statistics['n_instances'] = n_instances_lab

        # get a copy of the arrays, containing only the required instances
        feature_depth_lab = self.feature_depth[idx]
        path_lengths_lab = self.path_lengths[idx]
        tree_performance_lab = self.tree_performance[idx]

        # gather statistics from the feature_depth array, for each class label
        # shape is instances, trees, features, so [:,:,fd]
        depth_counts = [np.unique(feature_depth_lab[:,:,fd][~np.isnan(feature_depth_lab[:,:,fd])], return_counts = True) for fd in range(self.n_features)]

        # number of times each feature node was visited
        statistics['n_node_traversals'] = np.array([np.nansum(dcz[1]) for dcz in depth_counts], dtype=np.float32)
        # number of times feature was a root node (depth == 0)
        statistics['n_root_traversals'] = np.array([depth_counts[dc][1][np.where(depth_counts[dc][0] == 0)][0] if depth_counts[dc][1][np.where(depth_counts[dc][0] == 0)] else 0 for dc in range(len(depth_counts))], dtype=np.float32)
        # number of times feature was a root-child (depth == 1)
        statistics['n_child_traversals'] = np.array([depth_counts[dc][1][np.where(depth_counts[dc][0] == 1)][0] if depth_counts[dc][1][np.where(depth_counts[dc][0] == 1)] else 0 for dc in range(len(depth_counts))], dtype=np.float32)
        # number of times feature was a lower node (depth > 1)
        statistics['n_lower_traversals'] = np.array([np.nansum(depth_counts[dc][1][np.where(depth_counts[dc][0] > 1)] if any(depth_counts[dc][1][np.where(depth_counts[dc][0] > 1)]) else 0) for dc in range(len(depth_counts))], dtype=np.float32)
        # number of times feature was not a root
        statistics['n_nonroot_traversals'] = statistics['n_node_traversals'] - statistics['n_root_traversals'] # total feature visits - number of times feature was a root

        # number of correct predictions
        statistics['n_correct_preds'] = np.sum(tree_performance_lab) # total number of correct predictions
        statistics['n_path_length'] = np.sum(path_lengths_lab) # total path length accumulated by each feature

        # above measures normalised over all features
        p_ = lambda x : x / np.nansum(x)

        statistics['p_node_traversals'] = p_(statistics['n_node_traversals'])
        statistics['p_root_traversals'] = p_(statistics['n_root_traversals'])
        statistics['p_nonroot_traversals'] = p_(statistics['n_nonroot_traversals'])
        statistics['p_child_traversals'] = p_(statistics['n_child_traversals'])
        statistics['p_lower_traversals'] = p_(statistics['n_lower_traversals'])
        statistics['p_correct_preds'] = np.mean(tree_performance_lab) # accuracy

        statistics['m_node_traversals'] = np.mean(np.sum(~np.isnan(feature_depth_lab), axis = 1), axis = 0) # mean number of times feature appeared over all instances
        statistics['m_root_traversals'] = np.mean(np.sum(feature_depth_lab == 0, axis = 1), axis = 0) # mean number of times feature appeared as a root node, over all instances
        statistics['m_nonroot_traversals'] = np.mean(np.sum(np.nan_to_num(feature_depth_lab) > 0, axis = 1), axis = 0)
        statistics['m_child_traversals'] = np.mean(np.sum(np.nan_to_num(feature_depth_lab) == 1, axis = 1), axis = 0)
        statistics['m_lower_traversals'] = np.mean(np.sum(np.nan_to_num(feature_depth_lab) > 1, axis = 1), axis = 0)
        statistics['m_feature_depth'] = np.mean(np.nanmean(feature_depth_lab, axis = 1), axis = 0) # mean depth of each feature when it appears
        statistics['m_path_length'] = np.mean(np.nanmean(path_lengths_lab, axis = 1), axis = 0) # mean path length of each instance in the forest
        statistics['m_correct_preds'] = np.mean(np.mean(tree_performance_lab, axis = 1)) # mean prop. of trees voting correctly per instance

        if n_instances_lab > 1: # can't compute these on just one example
            statistics['sd_node_traversals'] = np.std(np.sum(~np.isnan(feature_depth_lab), axis = 1), axis = 0, ddof = 1) # sd of number of times... over all instances and trees
            statistics['sd_root_traversals'] = np.std(np.sum(feature_depth_lab == 0, axis = 1), axis = 0, ddof = 1) # sd of number of times feature appeared as a root node, over all instances
            statistics['sd_nonroot_traversals'] = np.std(np.sum(np.nan_to_num(feature_depth_lab) > 0, axis = 1), axis = 0, ddof = 1) # sd of number of times feature appeared as a nonroot node, over all instances
            statistics['sd_child_traversals'] = np.std(np.sum(np.nan_to_num(feature_depth_lab) == 1, axis = 1), axis = 0, ddof = 1)
            statistics['sd_lower_traversals'] = np.std(np.sum(np.nan_to_num(feature_depth_lab) > 1, axis = 1), axis = 0, ddof = 1)
            statistics['sd_feature_depth'] = np.std(np.nanmean(feature_depth_lab, axis = 1), axis = 0, ddof = 1) # sd depth of each feature when it appears
            statistics['sd_path_length'] = np.std(np.nanmean(path_lengths_lab, axis = 1), axis = 0, ddof = 1)
            statistics['sd_correct_preds'] = np.std(np.mean(tree_performance_lab, axis = 1), ddof = 1) # std prop. of trees voting correctly per instance
            statistics['se_node_traversals'] = sem(np.sum(~np.isnan(feature_depth_lab), axis = 1), axis = 0, ddof = 1, nan_policy = 'omit') # se of mean number of times feature appeared over all instances
            statistics['se_root_traversals'] = sem(np.sum(feature_depth_lab == 0, axis = 1), axis = 0, ddof = 1, nan_policy = 'omit') # se of mean of number of times feature appeared as a root node, over all instances
            statistics['se_nonroot_traversals'] = sem(np.sum(np.nan_to_num(feature_depth_lab) > 0, axis = 1), axis = 0, ddof = 1, nan_policy = 'omit') # sd of number of times feature appeared as a nonroot node, over all instances
            statistics['se_child_traversals'] = sem(np.sum(np.nan_to_num(feature_depth_lab) == 1, axis = 1), axis = 0, ddof = 1, nan_policy = 'omit')
            statistics['se_lower_traversals'] = sem(np.sum(np.nan_to_num(feature_depth_lab) > 1, axis = 1), axis = 0, ddof = 1, nan_policy = 'omit')
            statistics['se_feature_depth'] = sem(np.nanmean(feature_depth_lab, axis = 1), axis = 0, ddof = 1, nan_policy = 'omit') # se depth of each feature when it appears
            statistics['se_path_length'] = sem(np.nanmean(path_lengths_lab, axis = 1), axis = 0, ddof = 1, nan_policy = 'omit')
            statistics['se_correct_preds'] = sem(np.mean(tree_performance_lab, axis = 1), ddof = 1, nan_policy = 'omit') # se prop. of trees voting correctly per instance
        else:
            statistics['sd_node_traversals'] = np.full(self.n_features, np.nan)
            statistics['sd_root_traversals'] = np.full(self.n_features, np.nan)
            statistics['sd_nonroot_traversals'] = np.full(self.n_features, np.nan)
            statistics['sd_child_traversals'] = np.full(self.n_features, np.nan)
            statistics['sd_lower_traversals'] = np.full(self.n_features, np.nan)
            statistics['sd_feature_depth'] = np.full(self.n_features, np.nan)
            statistics['sd_path_length'] = np.full(self.n_features, np.nan)
            statistics['sd_correct_preds'] = np.full(self.n_features, np.nan)
            statistics['se_node_traversals'] = np.full(self.n_features, np.nan)
            statistics['se_root_traversals'] = np.full(self.n_features, np.nan)
            statistics['se_nonroot_traversals'] = np.full(self.n_features, np.nan)
            statistics['se_child_traversals'] = np.full(self.n_features, np.nan)
            statistics['se_lower_traversals'] = np.full(self.n_features, np.nan)
            statistics['se_feature_depth'] = np.full(self.n_features, np.nan)
            statistics['se_path_length'] = np.full(self.n_features, np.nan)
            statistics['se_correct_preds'] = np.full(self.n_features, np.nan)
        return(statistics)

    def forest_stats(self, class_labels = None):

        statistics = {}

        if class_labels is None:
            class_labels = np.unique(self.labels)
        for cl in class_labels:
            statistics[cl] = self.forest_stats_by_label(cl)

        statistics['all_classes'] = self.forest_stats_by_label()
        return(statistics)

    def major_class_from_paths(self, batch_idx, return_counts=False): # needs to be weighted
        pred_classes = [self.path_detail[batch_idx][p]['pred_class'] for p in range(len(self.path_detail[batch_idx]))]

        unique, counts = np.unique(pred_classes, return_counts=True)

        if return_counts:
            return(unique[np.argmax(counts)], dict(zip(unique, counts)))
        else: return(unique[np.argmax(counts)])
