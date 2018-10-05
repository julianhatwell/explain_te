import numpy as np
from CHIRPS import p_count, p_count_corrected
from collections import deque
from scipy import sparse

def as_tree_walk(tree_idx, instances, labels,
                instance_ids, n_instances,
                tree_pred, tree_pred_labels,
                tree_pred_proba, tree_correct,
                feature, threshold, path, features):

    # object for the results
    instance_paths = [{}] * n_instances

    # rare case that tree is a single node stub
    if len(feature) == 1:
        for ic in range(n_instances):
            if labels is None:
                true_class = None
            else:
                true_class = labels.values[ic]
            instance_paths[ic] = {'instance_id' : instance_ids[ic]
                                    , 'pred_class' : tree_pred[ic].astype(np.int64)
                                    , 'pred_class_label' : tree_pred_labels[ic]
                                    , 'pred_proba' : tree_pred_proba[ic].tolist()
                                    , 'true_class' : true_class
                                    , 'tree_correct' : tree_correct[ic]
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
                    true_class = None
                else:
                    true_class = labels.values[ic]
                instance_paths[ic] = {'instance_id' : instance_ids[ic]
                                        , 'pred_class' : tree_pred[ic].astype(np.int64)
                                        , 'pred_class_label' : tree_pred_labels[ic]
                                        , 'pred_proba' : tree_pred_proba[ic].tolist()
                                        , 'true_class' : true_class
                                        , 'tree_correct' : tree_correct[ic]
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
                instance_paths[ic]['path']['feature_idx'].append(feature[p])
                instance_paths[ic]['path']['feature_name'].append(feature_name)
                instance_paths[ic]['path']['feature_value'].append(feature_value)
                instance_paths[ic]['path']['threshold'].append(threshold[p])
                instance_paths[ic]['path']['leq_threshold'].append(leq_threshold)

    return(tree_idx, instance_paths)
