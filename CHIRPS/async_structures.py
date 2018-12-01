# this module is required for parallel processing
# parallel requires functions/classes to be in __main__ or already referenced.
# as this code is quite complex, the latter is preferred
import math
import numpy as np
from scipy import sparse
from copy import deepcopy
from collections import deque
from scipy.stats import chi2_contingency

# parallelisable function for the forest_walker class
def as_tree_walk(tree_idx, instances, labels, n_instances,
                tree_pred, tree_pred_labels,
                tree_pred_proba, tree_agree_maj_vote,
                feature, threshold, path, features):

    # object for the results
    tree_paths = [{}] * n_instances

    # rare case that tree is a single node stub
    if len(feature) == 1:
        for ic in range(n_instances):
            if labels is None:
                pred_class = None
            else:
                pred_class = labels[ic]
            tree_paths[ic] = {'pred_class' : tree_pred[ic].astype(np.int64)
                                    , 'pred_class_label' : tree_pred_labels[ic]
                                    , 'pred_proba' : tree_pred_proba[ic].tolist()
                                    , 'forest_pred_class' : pred_class
                                    , 'agree_maj_vote' : tree_agree_maj_vote[ic]
                                    , 'path' : {'feature_idx' : []
                                                            , 'feature_name' : []
                                                            , 'feature_value' : []
                                                            , 'threshold' : []
                                                            , 'leq_threshold' : []
                                                }
                                    }
    # usual case
    else:
        path_deque = deque(path)
        ic = -1 # instance_count
        while len(path_deque) > 0:
            p = path_deque.popleft()
            if feature[p] < 0: # leaf node
                continue
            pass_test = True
            if features is None:
                feature_name = None
            else:
                feature_name = features[feature[p]]
            if p == 0: # root node
                ic += 1
                feature_value = instances[ic, [feature[p]]].item(0)
                leq_threshold = feature_value <= threshold[p]
                if labels is None:
                    pred_class = None
                else:
                    pred_class = labels[ic]
                tree_paths[ic] = {'pred_class' : tree_pred[ic].astype(np.int64)
                                        , 'pred_class_label' : tree_pred_labels[ic]
                                        , 'pred_proba' : tree_pred_proba[ic].tolist()
                                        , 'forest_pred_class' : pred_class
                                        , 'agree_maj_vote' : tree_agree_maj_vote[ic]
                                        , 'path' : {'feature_idx' : [feature[p]]
                                                                , 'feature_name' : [feature_name]
                                                                , 'feature_value' : [feature_value]
                                                                , 'threshold' : [threshold[p]]
                                                                , 'leq_threshold' : [leq_threshold]
                                                    }
                                        }
            else:
                feature_value = instances[ic, [feature[p]]].item(0)
                leq_threshold = feature_value <= threshold[p]
                tree_paths[ic]['path']['feature_idx'].append(feature[p])
                tree_paths[ic]['path']['feature_name'].append(feature_name)
                tree_paths[ic]['path']['feature_value'].append(feature_value)
                tree_paths[ic]['path']['threshold'].append(threshold[p])
                tree_paths[ic]['path']['leq_threshold'].append(leq_threshold)

    return(tree_idx, tree_paths)

def as_CHIRPS(c_runner, target_class,
                sample_instances, sample_labels, forest,
                support_paths=0.1, alpha_paths=0.0,
                disc_path_bins=4, disc_path_eqcounts=False,
                score_func=1, weighting='chisq',
                algorithm='greedy_stab', bootstraps=0, delta=0.1,
                precis_threshold=0.95, batch_idx=None):
    # these steps make up the CHIRPS process:
    # mine paths for freq patts
    # fp growth mining
    c_runner.mine_path_segments(support_paths,
                            disc_path_bins, disc_path_eqcounts)

    # score and sort
    c_runner.score_sort_path_segments(sample_instances, sample_labels,
                                    target_class, alpha_paths, score_func, weighting)

    # greedily add terms to create rule
    c_runner.merge_rule(sample_instances=sample_instances,
                sample_labels=sample_labels,
                forest=forest,
                algorithm=algorithm,
                bootstraps=bootstraps,
                delta=delta,
                precis_threshold=precis_threshold)

    CHIRPS_exp = c_runner.get_CHIRPS_explainer()

    return(batch_idx, CHIRPS_exp)
