import sys
import math
import multiprocessing as mp
# import traceback
import numpy as np
from pandas import DataFrame, Series
from CHIRPS import p_count_corrected, if_nexists_make_dir, chisq_indep_test, entropy_corrected, contingency_test, confidence_weight
from pyfpgrowth import find_frequent_patterns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy import sparse
from scipy.stats import sem
from collections import defaultdict
from operator import itemgetter
from itertools import chain, repeat
from CHIRPS import config as cfg
from CHIRPS.async_structures import *

import warnings

class default_encoder(object):

    def transform(x):
        return(sparse.csr_matrix(x))
    def fit(x):
        return(x)

# this is inherited by CHIRPS_runner and data_container
class non_deterministic(object):

    def __init__(self, random_state=None):
        if random_state is None:
            self.random_state = 123
        else:
            self.random_state = random_state

    def set_random_state(self, random_state=None):
        if random_state is not None:
            self.random_state = random_state

    def default_if_none_random_state(self, random_state=None):
        if random_state is None:
            return(self.random_state)
        else:
            return(random_state)

# convenience class with more than just train_test_split
class data_split_container(object):

    def __init__(self, X_train, X_train_enc,
                X_test, X_test_enc,
                y_train, y_test,
                train_prior, test_prior,
                train_index=None, test_index=None):
        self.X_train = X_train
        self.X_train_enc = X_train_enc
        self.X_test_enc = X_test_enc
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_prior = train_prior
        self.test_prior = test_prior

        self.X_train_matrix = np.matrix(X_train)
        self.X_test_matrix = np.matrix(X_test)

        to_mat = lambda x : x.todense() if isinstance(x, sparse.csr.csr_matrix) \
                                        else x
        self.X_train_enc_matrix = to_mat(X_train_enc)
        self.X_test_enc_matrix = to_mat(X_test_enc)

        if train_index is None:
            self.train_index = y_train.index
        else:
            self.train_index = train_index
        if test_index is None:
            self.test_index = y_test.index
        else:
            self.test_index = test_index
        self.current_row_train = 0
        self.current_row_test = 0

    def get_which_split(self, which_split):
        # general getter for code re-use
        if which_split == 'test':
            instances = self.X_test
            instances_matrix = self.X_test_matrix
            instances_enc = self.X_test_enc
            instances_enc_matrix = self.X_test_enc_matrix
            labels = self.y_test
        else:
            instances = self.X_train
            instances_matrix = self.X_train_matrix
            instances_enc = self.X_train_enc
            instances_enc_matrix = self.X_train_enc_matrix
            labels = self.y_train
        return(instances, instances_matrix, instances_enc, instances_enc_matrix, labels)

    def get_by_id(self, instance_idx, which_split=None):

        if which_split is None:
            if all([True if i in self.y_test.index else False for i in instance_idx]):
                which_split = 'test'
            elif all([True if i in self.y_train.index else False for i in instance_idx]):
                which_split = 'train'
            else:
                print('ids found in neither or both partitions. Must be from a single partition.')
                return(None, None, None, None)

        instances, instances_matrix, instances_enc, instances_enc_matrix, labels = \
                                            self.get_which_split(which_split)

        # filter by the list of indices given
        instances = instances.loc[instance_idx]
        loc_index = [i for i, idx in enumerate(labels.index) if idx in instance_idx]
        instances_matrix = instances_matrix[loc_index,:]
        instances_enc = instances_enc[loc_index,:]
        instances_enc_matrix = instances_enc_matrix[loc_index,:]
        labels = labels[instance_idx]

        return(instances, instances_matrix, instances_enc, instances_enc_matrix, labels)

    def get_next(self, n_instances = 1, which_split='train'):

        instances, instances_matrix, instances_enc, instances_enc_matrix, labels = \
                                            self.get_which_split(which_split)

        if which_split == 'test':
            current_row = self.current_row_test
            self.current_row_test += n_instances
        else:
            current_row = self.current_row_train
            self.current_row_train += n_instances

        instances = instances[current_row:current_row + n_instances]
        instances_matrix = instances_matrix[current_row:current_row + n_instances]
        instances_enc = instances_enc[current_row:current_row + n_instances]
        instances_enc_matrix = instances_enc_matrix[current_row:current_row + n_instances]
        labels = labels[current_row:current_row + n_instances]

        return(instances, instances_matrix, instances_enc, instances_enc_matrix, labels)

    # leave one out by instance_id and encode the rest
    def get_loo_instances(self, instance_id, which_split='test'):

        instances, instances_matrix, instances_enc, instances_enc_matrix, labels = \
                                            self.get_which_split(which_split)

        if instance_id in instances.index:
            instances = instances.drop(instance_id)
            loc_index = labels.index == instance_id
            instances_matrix = instances_matrix[~loc_index,:]
            instances_enc = instances_enc[~loc_index,:]
            instances_enc_matrix = instances_enc_matrix[~loc_index,:]
            labels = labels.drop(instance_id)
            return(instances, instances_matrix, instances_enc, instances_enc_matrix, labels)
        else:
            print('Instance not found in this data partition')
            return(None, None, None, None, None)

    def to_dict(self):
        return({'X_train': self.X_train,
            'X_train_matrix' : self.X_train_matrix,
            'X_train_enc' : self.X_train_enc,
            'X_train_enc_matrix' : self.X_train_enc_matrix,
            'X_test' : self.X_test,
            'X_test_matrix' : self.X_test_matrix,
            'X_test_enc' : self.X_test_enc,
            'X_test_enc_matrix' : self.X_test_enc_matrix,
            'y_train' : self.y_train,
            'y_test' : self.y_test,
            'train_prior' : self.train_prior,
            'test_prior' : self.test_prior,
            'train_index' : self.train_index,
            'test_index' : self.test_index})

    def indexes_to_csv(self, save_path = ''):
        if_nexists_make_dir(save_path)
        Series(self.train_index).to_csv(path_or_buf = save_path + 'train_index.csv', index=False, header=False)
        Series(self.test_index).to_csv(path_or_buf = save_path + 'test_index.csv', index=False, header=False)

    def data_to_csv(self, save_path = '', encoded_features = None):
        if_nexists_make_dir(save_path)
        self.X_train.to_csv(path_or_buf = save_path + 'X_train.csv', header=False)
        if encoded_features:
            DataFrame(self.X_train_enc.todense(),
                    columns=encoded_features).to_csv(
                                                    path_or_buf = save_path + 'X_train_enc.csv',
                                                    header=False)
            DataFrame(self.X_test_enc.todense(),
                    columns=encoded_features).to_csv(
                                                    path_or_buf = save_path + 'X_test_enc.csv',
                                                    header=False)
        self.y_train.to_csv(path_or_buf = save_path + 'y_train.csv', header=False)
        self.X_test.to_csv(path_or_buf = save_path + 'X_test.csv', header=False)
        self.y_test.to_csv(path_or_buf = save_path + 'y_test.csv', header=False)

    def to_csv(self, save_path = '', encoded_features = None):
        self.data_to_csv(save_path = save_path, encoded_features = encoded_features)
        self.indexes_to_csv(save_path = save_path)

    def test_train_split(self): # behave as scikit-learn
        return(self.X_train, self.X_test, self.y_train, self.y_test)

# wrapper for data for convenience
class data_container(non_deterministic):

    def __init__(self
    , data
    , class_col
    , var_names = None
    , var_types = None
    , project_dir = None
    , save_dir = ''
    , random_state = None
    , spiel = ''):
        super().__init__(random_state)
        self.spiel = spiel
        self.data = data
        self.data_pre = DataFrame.copy(self.data)
        self.class_col = class_col
        self.save_dir = save_dir

        if project_dir is None:
            self.project_dir = cfg.project_dir
        else:
            self.project_dir = project_dir

        if var_names is None:
            self.var_names = list(self.data.columns)
        else:
            self.var_names = var_names

        if var_types is None:
            self.var_types = ['nominal' if dt.name == 'object' else 'continuous' for dt in self.data.dtypes.values]
        else:
            self.var_types = var_types

        self.features = [vn for vn in self.var_names if vn != self.class_col]
        self.class_names = list(self.data[self.class_col].unique())

        self.le_dict = {}
        self.var_dict = {}
        self.var_dict_enc = {}

        for i, (v, t) in enumerate(zip(self.var_names, self.var_types)):
            if t == 'nominal':
                # create a label encoder for all categoricals
                self.le_dict[v] = LabelEncoder().fit(self.data[v].unique())
                # create a dictionary of categorical names
                names = list(self.le_dict[v].classes_)
                # transform each categorical column
                self.data_pre[v] = self.le_dict[v].transform(self.data[v])
                # create the reverse lookup
                for n in names:
                    self.var_dict_enc[v + '_' + str(n)] = v
            else:
                self.data_pre[v] = self.data[v]

            self.var_dict[v] = {'labels' : names if t == 'nominal' else None
                                , 'labels_enc' : [v + '_' + str(n) for n in names] if t == 'nominal' else None
                                , 'class_col' : True if v == class_col else False
                                , 'data_type' : t
                                , 'order_col' : i}

        if any(n == 'nominal' for n in self.var_types ): del names
        del t

        self.categorical_features=[i for i, (c, t) in enumerate(zip([self.var_dict[f]['class_col'] for f in self.features],
        [self.var_dict[f]['data_type'] == 'nominal' for f in self.features])) if not c and t]

        # creates a flat list just for the features
        self.features_enc = []
        self.continuous_features = []
        for f, t in zip(self.var_names, self.var_types):
            if f == self.class_col: continue
            if t == 'continuous':
                self.continuous_features.append(f)
            else:
                self.features_enc.append(self.var_dict[f]['labels_enc'])

        # They get stuck on the end by encoding
        self.features_enc.append(self.continuous_features)
        # flatten out the nesting
        self.features_enc = list(chain.from_iterable(self.features_enc))

        # one hot encoding required for classifier
        # otherwise integer vectors will be treated as ordinal
        # OneHotEncoder takes an integer list as an argument to state which columns to encode
        # If no nominal vars, then simply convert to sparse matrix format
        if len(self.categorical_features) > 0:
            encoder = OneHotEncoder(categorical_features=self.categorical_features)
            encoder.fit(self.data_pre.values)
            self.encoder = encoder
        else:
            self.encoder = default_encoder

    # helper function for saving files
    def get_save_path(self, filename = ''):
        if len(self.project_dir) > 0:
            return(self.project_dir + cfg.path_sep + self.save_dir + cfg.path_sep + filename)
        else:
            return(self.save_dir + cfg.path_sep + filename)

    # helper function for data frame str / summary
    def rstr(self):
        return(self.data.shape, self.data.apply(lambda x: [x.unique()]))

    # a function to return any code from a label
    def get_code(self, col, label):
        if len(self.le_dict.keys()) > 0 and label in self.le_dict.keys():
            return self.le_dict[col].transform([label])[0]
        else:
            return(label)

    # a function to return any label from a code
    def get_label(self, col, label):
        if len(self.le_dict.keys()) > 0 and col in self.le_dict.keys():
            return self.le_dict[col].inverse_transform(label)
        else:
            return(label)

    # generate indexes for manual tt_split
    def get_tt_split_idx(self, test_size=0.3, random_state=None, shuffle=True):
        # common default setting: see class non_deterministic
        random_state = self.default_if_none_random_state(random_state)
        n_instances = self.data.shape[0]
        np.random.seed(random_state)
        test_idx = np.random.choice(n_instances - 1, # zero base
                                    size = round(test_size * n_instances),
                                    replace=False)
        # this method avoids scanning the array for each test_idx to find all the others
        train_pos = Series([True] * n_instances)
        train_pos.loc[test_idx] = False
        train_idx = np.array(train_pos.index[train_pos], dtype=np.int32)
        # train are currently in given order, the test are not
        if shuffle:
            np.random.shuffle(train_idx)

        return(train_idx, test_idx)

    def tt_split(self, train_index=None, test_index=None, test_size=0.3, random_state=None):

        # data in readiness
        X, y = self.data_pre[self.features], self.data_pre[self.class_col]

        # determine which method to use
        if train_index is None or test_index is None:
            # use scikit
            # common default setting: see class non_deterministic
            random_state = self.default_if_none_random_state(random_state)
            X, y = self.data_pre[self.features], self.data_pre[self.class_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        else:
            # use given indices
            X_test = X.loc[test_index]
            y_test = y.loc[test_index]
            X_train = X.loc[train_index]
            y_train = y.loc[train_index]

        # determine the prior for each split
        train_prior = y_train.value_counts().sort_index()/len(y_train)
        test_prior = y_test.value_counts().sort_index()/len(y_test)

        # create an encoded copy
        X_train_enc = self.encoder.transform(X_train)
        X_test_enc = self.encoder.transform(X_test)

        # coo format needs to be converted as it has limited properties
        to_csr = lambda x : x.tocsr() if isinstance(x, sparse.coo.coo_matrix) \
                                        else x
        X_train_enc = to_csr(X_train_enc)
        X_test_enc = to_csr(X_test_enc)

        tt = data_split_container(X_train = X_train,
        X_train_enc = X_train_enc,
        X_test = X_test,
        X_test_enc = X_test_enc,
        y_train = y_train,
        y_test = y_test,
        train_prior = train_prior,
        test_prior = test_prior)

        return(tt)

    def get_meta(self):
        return({'class_col' : self.class_col,
                'class_names' : self.class_names,
                'class_names_label_order' : self.get_label(self.class_col, \
                            [i for i in range(len(self.class_names))]),
                'var_names' : self.var_names,
                'var_types' : self.var_types,
                'features' : self.features,
                'features_enc' : self.features_enc,
                'var_dict' : self.var_dict,
                'var_dict_enc' : self.var_dict_enc,
                'categorical_features' : self.categorical_features,
                'continuous_features' : self.continuous_features,
                'le_dict' : self.le_dict,
                'get_label' : self.get_label,
                'random_state' : self.random_state,
                'get_save_path' : self.get_save_path
                })

# classes and functions for the parallelisable tree_walk
class batch_paths_container(object):
    def __init__(self,
    target_classes,
    path_detail,
    by_tree):
        self.target_classes = target_classes
        self.by_tree = by_tree # given as initial state, not used to reshape on construction. use flip() if necessary
        self.path_detail = path_detail

    def flip(self):
        n = len(self.path_detail[0])
        flipped_paths = [[]] * n
        for i in range(n):
            flipped_paths[i] =  [p[i] for p in self.path_detail]
        self.path_detail = flipped_paths
        self.by_tree = not self.by_tree

    def target_class_from_paths(self, batch_idx):
        return(self.target_classes[batch_idx])

    def major_class_from_paths(self, batch_idx, return_counts=False):
        if self.by_tree:
            pred_classes = [self.path_detail[p][batch_idx]['pred_class'] for p in range(len(self.path_detail))]
        else:
            pred_classes = [self.path_detail[batch_idx][p]['pred_class'] for p in range(len(self.path_detail[batch_idx]))]

        unique, counts = np.unique(pred_classes, return_counts=True)

        if return_counts:
            return(unique[np.argmax(counts)], dict(zip(unique, counts)))
        else: return(unique[np.argmax(counts)])

    def get_CHIRPS_runner(self, batch_idx, meta_data, which_trees='majority', random_state=123, feature_values=True):

        batch_idx = math.floor(batch_idx) # make sure it's an integer
        true_to_lt = lambda x: '<' if x == True else '>'

        # extract the paths we want by filtering on tree performance
        if self.by_tree:
            n_paths = len(self.path_detail)
            if which_trees == 'correct':
                paths_info, paths_weights, paths_pred_proba = [i for i in map(list, zip(*[itemgetter('path', 'estimator_weight', 'pred_proba')(self.path_detail[batch_idx][pd]) for pd in range(n_paths) if self.path_detail[pd][batch_idx]['agree_maj_vote']]))]
            elif which_trees == 'majority':
                major_class = self.major_class_from_paths(batch_idx, return_counts=False)
                paths_info, paths_weights, paths_pred_proba = [i for i in map(list, zip(*[itemgetter('path', 'estimator_weight', 'pred_proba')(self.path_detail[batch_idx][pd]) for pd in range(n_paths) if self.path_detail[pd][batch_idx]['pred_class'] == major_class]))]
            elif which_trees == 'minority':
                major_class = self.major_class_from_paths(batch_idx, return_counts=False)
                paths_info, paths_weights, paths_pred_proba = [i for i in map(list, zip(*[itemgetter('path', 'estimator_weight', 'pred_proba')(self.path_detail[batch_idx][pd]) for pd in range(n_paths) if self.path_detail[pd][batch_idx]['pred_class'] != major_class]))]
            else:
                paths_info, paths_weights, paths_pred_proba = [i for i in map(list, zip(*[itemgetter('path', 'estimator_weight', 'pred_proba')(self.path_detail[batch_idx][pd]) for pd in range(n_paths) for pd in range(n_paths)]))]
        else:
            n_paths = len(self.path_detail[batch_idx])
            if which_trees == 'correct':
                paths_info, paths_weights, paths_pred_proba = [i for i in map(list, zip(*[itemgetter('path', 'estimator_weight', 'pred_proba')(self.path_detail[batch_idx][pd]) for pd in range(n_paths) if self.path_detail[batch_idx][pd]['agree_maj_vote']]))]
            elif which_trees == 'majority':
                major_class = self.major_class_from_paths(batch_idx, return_counts=False)
                paths_info, paths_weights, paths_pred_proba = [i for i in map(list, zip(*[itemgetter('path', 'estimator_weight', 'pred_proba')(self.path_detail[batch_idx][pd]) for pd in range(n_paths) if self.path_detail[batch_idx][pd]['pred_class'] == major_class]))]
            elif which_trees == 'minority':
                major_class = self.major_class_from_paths(batch_idx, return_counts=False)
                paths_info, paths_weights, paths_pred_proba = [i for i in map(list, zip(*[itemgetter('path', 'estimator_weight', 'pred_proba')(self.path_detail[batch_idx][pd]) for pd in range(n_paths) if self.path_detail[batch_idx][pd]['pred_class'] != major_class]))]
            else:
                paths_info, paths_weights, paths_pred_proba = [i for i in map(list, zip(*[itemgetter('path', 'estimator_weight', 'pred_proba')(self.path_detail[batch_idx][pd]) for pd in range(n_paths)]))]

        # support for SAMME.R


        # path formatting - should it be on values level or features level
        if feature_values:
            paths = [[]] * len(paths_info)
            for i, p in enumerate(paths_info):
                paths[i] = [(f, leq, t) for f, leq, t in zip(p['feature_name'], p['leq_threshold'], p['threshold'])]
        else:
            paths = [p['feature_name'] for p in paths_info]

        # per tree performance stats for the whole ensemble (not filtered)
        if self.by_tree:
            tree_preds, estimator_weights, pred_probas = [i for i in map(list, zip(*[itemgetter('pred_class', 'estimator_weight', 'pred_proba')(self.path_detail[t][batch_idx]) for t in range(n_paths)]))]
        else:
            tree_preds, estimator_weights, pred_probas = [i for i in map(list, zip(*[itemgetter('pred_class', 'estimator_weight', 'pred_proba')(self.path_detail[batch_idx][t]) for t in range(n_paths)]))]

        # simply determine the count and proportion of trees according to votes
        model_votes = p_count_corrected(tree_preds, [i for i in range(len(meta_data['class_names']))], weights=estimator_weights)

        # determine the weighted count of trees  (support for SAMME.R)
        # based on SAMME.R quantities
        # confidence_weights = confidence_weight(np.array(pred_probas), 'conf_weight')
        # confidence_weights = np.mean(confidence_weights + abs(confidence_weights.min(axis=1).reshape(-1, 1)), axis=0)
        # based on predicted probabilities
        confidence_weights = np.sum(pred_probas, axis=0)
        confidence_weights = p_count_corrected([i for i in range(len(meta_data['class_names']))], [i for i in range(len(meta_data['class_names']))], confidence_weights)

        # NOT DOING ANYTHING WITH PATH LEVEL PRED PROBA YET
        # return an object for requested instance
        c_runner = CHIRPS_runner(meta_data, paths, paths_weights, paths_pred_proba,
                                    tree_preds, model_votes, confidence_weights,
                                    self.major_class_from_paths(batch_idx),
                                    self.target_class_from_paths(batch_idx),
                                    random_state)
        return(c_runner)

class forest_walker(object):

    def __init__(self
    , forest
    , meta_data):
        self.forest = forest
        self.features = meta_data['features_enc']
        self.n_features = len(self.features)
        self.class_col = meta_data['class_col']
        self.n_classes = len(meta_data['class_names'])

        # weights for standard Boosted models
        if not hasattr(forest, 'estimator_weights_'):
            self.forest.estimator_weights_ = np.ones(len(forest.estimators_))

        meta_le_dict = meta_data['le_dict']
        meta_get_label = meta_data['get_label']

        if self.class_col in meta_le_dict.keys():
            self.get_label = meta_get_label
        else:
            self.get_label = None

        # base counts for all trees
        self.root_features = np.zeros(len(self.features)) # set up a 1d feature array to count features appearing as root nodes
        self.child_features = np.zeros(len(self.features))
        self.lower_features = np.zeros(len(self.features))

        # walk through each tree to get the structure
        for t, tree in enumerate(self.forest.estimators_):

            # root, child and lower counting, one time only (first class)
            structure = tree.tree_
            feature = structure.feature
            children_left = structure.children_left
            children_right = structure.children_right

            self.root_features[feature[0]] += 1
            if children_left[0] >= 0:
                self.child_features[feature[children_left[0]]] +=1
            if children_right[0] >= 0:
                self.child_features[feature[children_right[0]]] +=1

            for j, f in enumerate(feature):
                if j < 3: continue # root and children
                if f < 0: continue # leaf nodes
                self.lower_features[f] += 1

    def full_survey(self
        , instances
        , labels):

        self.instances = instances
        self.labels = labels
        self.n_instances = instances.shape[0]

        if labels is not None:
            if len(labels) != self.n_instances:
                raise ValueError("number of labels and instances does not match")

        trees = self.forest.estimators_
        self.n_trees = len(trees)

        self.feature_depth = np.full((self.n_instances, self.n_trees, self.n_features), np.nan) # set up a 1d feature array for counting
        self.tree_predictions = np.full((self.n_instances, self.n_trees), np.nan)
        self.tree_performance = np.full((self.n_instances, self.n_trees), np.nan)
        self.path_lengths = np.zeros((self.n_instances, self.n_trees))

        # walk through each tree
        for t, tree in enumerate(trees):
            # get the feature vector out of the tree object
            feature = tree.tree_.feature

            self.tree_predictions[:, t] = tree.predict(self.instances)
            self.tree_performance[:, t] = self.tree_predictions[:, t] == self.labels

            # extract path and get path lengths
            path = tree.decision_path(self.instances).indices
            paths_begin = np.where(path == 0)
            paths_end = np.append(np.where(path == 0)[0][1:], len(path))
            self.path_lengths[:, t] = paths_end - paths_begin

            depth = 0
            instance = -1
            for p in path:
                if feature[p] < 0: # leaf node
                    # TO DO: what's in a leaf node
                    continue
                if p == 0: # root node
                    instance += 1 # a new instance
                    depth = 0 # a new path
                else:
                    depth += 1 # same instance, descends tree one more node
                self.feature_depth[instance][t][feature[p]] = depth

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

    def tree_structures(self, tree, instances, labels, n_instances):

        # structural objects from tree
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        path = tree.decision_path(instances).indices

        # predictions from tree
        tree_pred = tree.predict(instances)
        tree_pred_proba = tree.predict_proba(instances)

        if labels is None:
            tree_agree_maj_vote = [None] * n_instances
        else:
            tree_agree_maj_vote = tree_pred == labels

        if labels is not None:
            tree_pred_labels = self.get_label(self.class_col, tree_pred.astype(int))
        else:
            tree_pred_labels = tree_pred

        return(tree_pred, tree_pred_labels, \
                tree_pred_proba, \
                tree_agree_maj_vote, feature, threshold, path)

    def forest_walk(self, instances, labels = None, forest_walk_async=False):

        features = self.features
        n_instances = instances.shape[0]

        if forest_walk_async:
            async_out = []
            n_cores = mp.cpu_count()-2
            pool = mp.Pool(processes=n_cores)

            for i, (t, est_wt) in enumerate(zip(self.forest.estimators_, self.forest.estimator_weights_)):

                # process the tree
                tree_pred, tree_pred_labels, \
                tree_pred_proba, \
                tree_agree_maj_vote, \
                feature, threshold, path = self.tree_structures(t, instances, labels, n_instances)
                # walk the tree
                async_out.append(pool.apply_async(as_tree_walk,
                                                (i, instances, labels, n_instances,
                                                tree_pred, tree_pred_labels,
                                                tree_pred_proba,
                                                tree_agree_maj_vote,
                                                feature, threshold, path, features, est_wt)
                                                ))

            # block and collect the pool
            pool.close()
            pool.join()

            # get the async results and sort to ensure original tree order and remove tree index
            tp = [async_out[j].get() for j in range(len(async_out))]
            tp.sort()
            tree_paths = [tp[k][1] for k in range(len(tp))]

        else:
            tree_paths = [[]] * len(self.forest.estimators_)
            for i, (t, est_wt) in enumerate(zip(self.forest.estimators_, self.forest.estimator_weights_)):

                # process the tree
                tree_pred, tree_pred_labels, \
                tree_pred_proba, \
                tree_agree_maj_vote, \
                feature, threshold, path = self.tree_structures(t, instances, labels, n_instances)
                # walk the tree
                _, tree_paths[i] = as_tree_walk(i, instances, labels, n_instances,
                                                tree_pred, tree_pred_labels,
                                                tree_pred_proba,
                                                tree_agree_maj_vote,
                                                feature, threshold, path, features, est_wt)

        return(batch_paths_container(labels, tree_paths, by_tree=True))

# classes and functions for the parallelisable CHIRPS algorithm

# this is to have the evaluator function inherited from one place
class evaluator(object):

    def evaluate(self, prior_labels, post_idx, class_names=None):

        if class_names is None:
            class_names = [i for i in range(len(np.unique(prior_labels)))]

        # priors
        all_c = len(prior_labels) # count all
        prior = p_count_corrected(prior_labels, class_names)

        # basic results
        p_counts = p_count_corrected(prior_labels[post_idx], class_names)
        counts = p_counts['counts'] # cc, covered and correct (for any label)
        c = np.sum(counts) # covered
        ci = c - counts # covered incorrect (total of other labels covered - won't add up to number of labels so do not sum ci)
        labels = p_counts['labels']
        posterior = p_counts['p_counts']

        # coverage
        # tp + fp / tp + fp + tn + fn
        coverage = c / all_c
        # xcov = tp + fp / tp + fp + tn + fn + current instance, laplace corrected
        xcoverage = (c + 1)/(all_c + len(class_names) + 1)

        # stab = tp / tp + fp + current instance, laplace corrected
        stability = (counts + 1) / (np.sum(counts) + len(class_names) + 1)

        # negative results
        np_counts = p_count_corrected(prior_labels[np.logical_not(post_idx)], class_names)
        ncounts = np_counts['counts'] # ncc, not covered but still correct (for any label)
        nc = all_c - c # not covered
        nci = np.sum(ncounts) - ncounts
        nposterior = np_counts['p_counts']
        # tn / tn + fn, (all labels - the ones predicted) / (all instances - covered instances)
        npv = nci /(nci + ci)

        # rfhc = [] # score from ForEx++ and rf+hc papers
        # for lab in labels:
        #     cc = counts[lab]
        #     ic = sum([c for c, l in zip(counts, labels) if l != lab])
        #     rfhc.append((cc - ic) / (cc + ic) + cc / (ic + 1))

        chisq = chisq_indep_test(counts, prior['counts'])[1] # p-value
        kl_div = entropy_corrected(posterior, prior['p_counts'])

        # TPR (recall) TP / (TP + FN)
        recall = counts / prior['counts']

        # F1
        p_corrected = np.array([p if p > 0.0 else 1.0 for p in posterior]) # to avoid div by zeros
        r_corrected = np.array([r if r > 0.0 else 1.0 for r in recall]) # to avoid div by zeros
        f1 = [2] * ((posterior * recall) / (p_corrected + r_corrected))

        not_covered_counts = counts + (np.sum(prior['counts']) - prior['counts']) - (np.sum(counts) - counts)
        # accuracy = (TP + TN) / num_instances formula: https://books.google.co.uk/books?id=ubzZDQAAQBAJ&pg=PR75&lpg=PR75&dq=rule+precision+and+coverage&source=bl&ots=Aa4Gj7fh5g&sig=6OsF3y4Kyk9KlN08OPQfkZCuZOc&hl=en&sa=X&ved=0ahUKEwjM06aW2brZAhWCIsAKHY5sA4kQ6AEIUjAE#v=onepage&q=rule%20precision%20and%20coverage&f=false
        accu = not_covered_counts/prior['counts'].sum()

        # to avoid div by zeros
        pri_corrected = np.array([pri if pri > 0.0 else 1.0 for pri in prior['p_counts']])
        pos_corrected = np.array([pos if pri > 0.0 else 0.0 for pri, pos in zip(prior['p_counts'], posterior)])
        if counts.sum() == 0:
            rec_corrected = np.zeros(len(pos_corrected))
            cov_corrected = np.ones(len(pos_corrected))
        else:
            rec_corrected = counts / counts.sum()
            cov_corrected = np.array([counts.sum() / prior['counts'].sum()])

        # lift = precis / (total_cover * prior)
        lift = pos_corrected / ( cov_corrected * pri_corrected )

        output = {'count_all' : all_c,
                'covered' : c,
                'not_covered' : nc,
                'cc' : counts,
                'ci' : ci,
                'ncc' : ncounts,
                'nci' : nci,
                'coverage' : coverage,
                'xcoverage' : xcoverage,
                'npv' : npv,
                'stability' : stability,
                'prior' : prior,
                'posterior' : posterior,
                'counts' : counts,
                'labels' : labels,
                'recall' : recall,
                'f1' : f1,
                'accuracy' : accu,
                'lift' : lift,
                'chisq' : chisq,
                'kl_div' : kl_div
                }
        return(output)

    def prettify_rule(self, rule=None, var_dict=None):

        if rule is None: # default
            rule = self.pruned_rule

        if var_dict is None: # default - match prediction model
            var_dict = self.var_dict_enc

        Tr_Fa = lambda x, y, z : x + ' False' if y else x + ' True'
        lt_gt = lambda x, y, z : x + ' <= ' + str(z) if y else x + ' > ' + str(z)
        def bin_or_cont(x, y, z):
            if x in var_dict:
                return(Tr_Fa(x,y,z))
            else:
                return(lt_gt(x,y,z))
        return(' AND '.join([bin_or_cont(f, t, v) for f, t, v in rule]))

# this is inherited by CHIRPS_explainer and CHIRPS_runner
class rule_evaluator(non_deterministic, evaluator):

    # allows new values other than those already in self
    def init_values(self, rule=None, features=None, class_names=None):

        # sub-classes must have these three properties
        if rule is not None:
            if rule == 'pruned':
                rule = self.pruned_rule
            else:
                rule = rule
        else:
            rule = self.rule

        if features is None:
            features = self.features_enc # default
        if class_names is None:
            class_names = self.class_names

        return(rule, features, class_names)

    def init_instances(self, instances=None, labels=None):
        # check presence of optional sample datasets:
        # train (or other) for optimisation of rule merge
        # test (or other) for evaluation of rule
        if instances is None:
            try:
                instances = self.sample_instances
            except AttributeError:
                print('Sample intances (e.g. X_train_enc) are required for rule evaluation')
                return(None, None)
        if labels is None:
            try:
                labels = self.sample_labels
            except AttributeError:
                print('No sample labels (e.g. y_train) were given. Do you plan to predict some new ones?')
                return(instances, None)

        return(instances, labels)

    def init_dicts(self, var_dict=None, var_dict_enc=None):

        if var_dict is None:
            try:
                var_dict = self.var_dict
            except AttributeError:
                print('Feature dictionary (meta data) required for rule evaluation')
                return(None, None)

        if var_dict_enc is None:
            try:
                var_dict_enc = self.var_dict_enc # can be None
            except AttributeError:
                return(instances, None)

        return(var_dict, var_dict_enc)

    # apply a rule on an instance space, returns covered instance idx
    def apply_rule(self, rule=None, instances=None, features=None):

        lt_gt = lambda x, y, z : x <= y if z else x > y # if z is True, x <= y else x > y
        idx = np.full(instances.shape[0], 1, dtype='bool')
        for r in rule:
            idx = np.logical_and(idx, lt_gt(instances.getcol(features.index(r[0])).toarray().flatten(), r[2], r[1]))
            if np.isnan(r[2]):
                print('nan:', str(rule))
        return(idx)
    # score a rule on an instance space
    def evaluate_rule(self, rule=None, features=None, class_names=None,
                        sample_instances=None, sample_labels=None, target_class=None):

        # allow new values or get self properties
        rule, features, class_names = self.init_values(rule=rule, features=features, class_names=class_names)
        instances, labels = self.init_instances(instances=sample_instances, labels=sample_labels)

        # get the covered idx
        idx = self.apply_rule(rule=rule, instances=instances, features=features)

        metrics = self.evaluate(prior_labels=labels, post_idx=idx)
        # collect metrics
        return(metrics)

    def categorise_rule_features(self, rule=None, var_dict=None, var_dict_enc=None):

        # allow new values or get self properties
        rule, _, _ = self.init_values(rule=rule)
        var_dict, var_dict_enc = self.init_dicts(var_dict=var_dict, var_dict_enc=var_dict_enc)

        # sort out features in a rule belonging to parent groups
        parent_features = {}
        for i, item in enumerate(rule):
            # nominal vars
            if item[0] in var_dict_enc:
                parent_item = var_dict_enc[item[0]]
                # with a single True value
                if item[1]: # True (less than thresh); is a disjoint set or one or more
                    if parent_item not in parent_features.keys():
                        parent_features.update({ parent_item : 'disjoint'}) # capture the parent feature
                # nominal vars with one or more False values (a disjoint rule);
                else: # False (greater than thresh); there can be only one for each parent
                    parent_features.update({ parent_item : item[0]}) # capture the parent feature and the child
            else: # continuous
                parent_features.update({item[0] : 'continuous'}) # capture the type of bound
        return(parent_features)

    def get_rule_complements(self, rule='pruned', var_dict=None, var_dict_enc=None):

        # allow new values or get self properties
        rule, _, _ = self.init_values(rule=rule)
        var_dict, var_dict_enc = self.init_dicts(var_dict=var_dict, var_dict_enc=var_dict_enc)

        parent_features = self.categorise_rule_features(rule=rule, var_dict=var_dict, var_dict_enc=var_dict_enc)
        rule_complements = {}

        for prnt in parent_features:
            if parent_features[prnt] == 'disjoint':
                where_false = np.where(np.array(var_dict[prnt]['upper_bound']) < 1)[0] # upper bound less than 1 is True
                if len(where_false) == 1: # one value can just be flipped to true
                    rule_complement = rule.copy()
                    for i, item in enumerate(rule):
                        if item[0] in var_dict[prnt]['labels_enc']:
                            rule_complement[i] = (item[0], False, item[2])
                            rule_complements.update({ prnt : rule_complement})
                else: # need to flip the disjoint set
                    rule_complement = []
                    for item in rule:
                        # keep only the other items
                        if item[0] in var_dict[prnt]['labels_enc']:
                            continue
                        else:
                            rule_complement.append(item)
                    # add the flipped disjoint set
                    for i, ub in enumerate(var_dict[prnt]['upper_bound']):
                        if ub >= 1:
                            rule_complement.append((var_dict[prnt]['labels_enc'][i], True, 0.5))
                    rule_complements.update({ prnt : rule_complement})
            elif parent_features[prnt] == 'continuous':
                rule_complement = rule.copy()
                for i, item in enumerate(rule):
                        if item[0] == prnt: # basic flip - won't work for both because rule eval is logical AND
                            rule_complement[i] = (item[0], not item[1], item[2])
                            rule_complements.update({ prnt : rule_complement})
            else: # its a single False (greater than thresh) - simple flip
                rule_complement = rule.copy()
                for i, item in enumerate(rule):
                    if item[0] == parent_features[prnt]:
                        rule_complement[i] = (item[0], True, item[2])
                        rule_complements.update({ prnt : rule_complement})

        return(rule_complements)

    def eval_rule_complements(self, sample_instances, sample_labels, rule_complements=None):

        # general setup
        instances, labels = self.init_instances(instances=sample_instances, labels=sample_labels)
        if rule_complements is None:
            rule_complements = self.get_rule_complements()

        rule_complement_results = []
        for feature in rule_complements:
            rc = rule_complements[feature]
            eval = self.evaluate_rule(rule=rc, sample_instances=instances, sample_labels=labels)
            kl_div = entropy_corrected(self.evaluate_rule(rule=self.rule, sample_instances=instances, sample_labels=labels)['posterior'], eval['posterior'])
            rule_complement_results.append( { 'feature' : feature,
                                            'rule' : rc,
                                            'pretty_rule' : self.prettify_rule(rc),
                                            'eval' :  eval,
                                            'kl_div' : kl_div } )

        return(rule_complement_results)

class CHIRPS_explainer(rule_evaluator):

    def __init__(self, random_state,
                features, features_enc, class_names,
                class_col, get_label,
                var_dict, var_dict_enc,
                paths, paths_weights, patterns,
                rule, pruned_rule,
                target_class, target_class_label,
                major_class, major_class_label,
                model_votes,
                confidence_weights,
                accumulated_weights,
                isolation_pos,
                posterior,
                stability,
                accuracy,
                counts,
                recall,
                f1,
                cc,
                ci,
                ncc,
                nci,
                npv,
                lift,
                chisq,
                kl_div,
                algorithm,
                elapsed_time):
        self.random_state = random_state
        self.features = features
        self.features_enc = features_enc
        self.class_names = class_names
        self.class_col = class_col
        self.get_label = get_label
        self.var_dict = var_dict
        self.var_dict_enc = var_dict_enc
        self.paths = paths
        self.paths_weights = paths_weights
        self.patterns = patterns
        self.rule = rule
        self.pruned_rule = pruned_rule
        self.target_class = target_class
        self.target_class_label = target_class_label
        self.major_class = major_class
        self.major_class_label = major_class_label
        self.model_votes = model_votes
        self.confidence_weights = confidence_weights
        self.accumulated_weights = accumulated_weights
        self.isolation_pos = isolation_pos
        self.posterior = posterior
        self.stability = stability
        self.accuracy = accuracy
        self.counts = counts
        self.recall = recall
        self.f1 = f1
        self.cc = cc
        self.ci = ci
        self.ncc = ncc
        self.nci = nci
        self.npv = npv
        self.lift = lift
        self.chisq = chisq
        self.kl_div = kl_div
        self.algorithm = algorithm
        self.elapsed_time = elapsed_time

        # instance meta data
        self.prior = self.posterior[0]
        self.forest_vote_share = self.model_votes['p_counts'][self.target_class]
        self.conf_weight_forest_vote_share = self.confidence_weights['p_counts'][self.target_class]
        remaining_values = self.model_votes['p_counts'][np.where(self.model_votes['p_counts'] != self.forest_vote_share)]
        second_greatest = remaining_values[np.argmax(remaining_values)]
        self.forest_vote_margin = self.forest_vote_share - second_greatest
        remaining_values = self.confidence_weights['p_counts'][np.where(self.confidence_weights['p_counts'] != self.conf_weight_forest_vote_share)]
        second_greatest = remaining_values[np.argmax(remaining_values)]
        self.conf_weight_forest_vote_margin = self.conf_weight_forest_vote_share - second_greatest

        self.pretty_rule = self.prettify_rule()
        self.rule_len = len(self.pruned_rule)

        # final metrics from rule merge step (usually based on training set)
        self.est_prec = list(reversed(self.posterior))[0][self.target_class]
        self.est_stab = list(reversed(self.stability))[0][self.target_class]
        self.est_recall = list(reversed(self.recall))[0][self.target_class]
        self.est_f1 = list(reversed(self.f1))[0][self.target_class]
        self.est_cc = list(reversed(self.cc))[0][self.target_class]
        self.est_ci = list(reversed(self.ci))[0][self.target_class]
        self.est_ncc = list(reversed(self.ncc))[0][self.target_class]
        self.est_nci = list(reversed(self.nci))[0][self.target_class]
        self.est_npv = list(reversed(self.npv))[0][self.target_class]
        self.est_acc = list(reversed(self.accuracy))[0][self.target_class]
        self.est_lift = list(reversed(self.lift))[0][self.target_class]
        self.est_coverage = list(reversed(self.counts))[0].sum() / self.counts[0].sum()
        self.est_xcoverage = list(reversed(self.counts))[0].sum() / (self.counts[0].sum() + 1)
        self.est_kl_div = list(reversed(self.kl_div))[0]
        self.posterior_counts = list(reversed(self.counts))[0]
        self.prior_counts = self.counts[0]

    def get_distribution_by_rule(self, sample_instances, size=None,
                                    rule='pruned', features=None,
                                    random_state=None):
        # take an instance and a sample instance set
        # return a distribution to match the sample set
        # mask any features not involved in the rule with the original instance

        # should usually get the feature list internally from init_values
        rule, features, _ = self.init_values(rule=rule, features=features)
        instances, _ = self.init_instances(instances=sample_instances)
        if size is None:
            size = instances.shape[0]

        # get instances covered by rule
        idx = self.apply_rule(rule=rule, instances=instances, features=features)

        instances = instances[idx]
        n_instances = instances.shape[0]

        # reproducibility
        random_state = self.default_if_none_random_state(random_state)

        # get a distribution for those instances covered by rule
        idx = np.random.choice(n_instances, size = size, replace=True)
        distributions = instances[idx]

        return(distributions)

    def mask_by_instance(self, instance, sample_instances, rule, feature,
                            features=None, var_dict=None, var_dict_enc=None,
                            size=None, random_state=None):

        # should usually get the feature list internally from init_values
        _, features, _ = self.init_values(rule=rule, features=features)
        var_dict, var_dict_enc = self.init_dicts(var_dict=var_dict, var_dict_enc=var_dict_enc)
        instances, _ = self.init_instances(instances=sample_instances)
        if size is None:
            size = instances.shape[0]

        try: # get a distribution given rule
            # first will contain a distribution of values for the feature that is reversed in the rule complement
            # the remaining features will be masked by the current instance
            rule_covered_dists = self.get_distribution_by_rule(instances,
                                                        size=size,
                                                        rule=rule,
                                                        features=None,
                                                        random_state=random_state)
            mask_cover = True

        except ValueError: # no coverage for rule comp - failed in method get_distribution_by_rule.
        # need to fall back to a distribution that doesn't respect joint distribution
            try: # flipped rule term only
                if var_dict[feature]['data_type'] == 'continuous':
                    flipped_rule_term = [item for item in rule if item[0] == feature]
                else:
                    flipped_rule_term = [item for item in rule if item[0] in var_dict[feature]['labels_enc']]
                # get a distribution given rule
                # first will contain a distribution of values for the feature that is reversed in the rule complement
                # the remaining features will be masked by the current instance
                rule_covered_dists = self.get_distribution_by_rule(instances,
                                                            size=size,
                                                            rule=flipped_rule_term,
                                                            features=None,
                                                            random_state=random_state)

            except ValueError: # couldn't create a suitable value outside the range of the sample instances - failed again in method get_distribution_by_rule.
                # if we've arrived here, then there are no instances in the provided sample that match the conditions in the flipped rule term.
                # assume this is an outlier continuous value.
                # in that case, we need to fake it - provide some values a little outside the extremity
                # first identify the boundary and whether greater or less than in required
                for j, f in enumerate(features):
                    if f == feature:
                        if var_dict[f]['data_type'] == 'continuous':
                            columnvec = instances[:, j].todense()
                            if np.issubdtype(columnvec.dtype, np.integer): # for integers, extend with integer series
                                for item in flipped_rule_term:
                                    if item[1]: # less than thresh
                                        distance_from_mean = columnvec.mean() - columnvec.min().absolute()
                                        delta = ((distance_from_mean * 3.1 / 3) - distance_from_mean).absolute() # we would like a bell curve where the third st.dev reaches 3.1 times the distance from mean
                                        dist = (item[2] - 0.5 - np.random.poisson(lam=delta, size=size))[:, np.newaxis] # generate some values
                                    else: # greater than
                                        distance_from_mean = columnvec.max() - columnvec.mean()
                                        delta = (distance_from_mean * 3.1 / 3) - distance_from_mean # we would like a bell curve where the third st.dev reaches 3.1 times the distance from mean
                                        dist = (item[2] + 0.5 + np.random.poisson(lam=delta, size=size))[:, np.newaxis] # generate some values
                            else: # for reals, extend with a continuous distribution
                                micro_diff = columnvec.min() - np.finfo(np.dtype(columnvec)).eps
                                for item in flipped_rule_term:
                                    if item[1]: # less than thresh
                                        distance_from_mean = columnvec.mean() - columnvec.min().absolute()
                                        delta = ((distance_from_mean * 3.1 / 3) - distance_from_mean).absolute() # we would like a bell curve where the third st.dev reaches 3.1 times the distance from mean
                                        dist = (micro_diff - np.random.gamma(shape=1, scale=delta, size=size))[:, np.newaxis] # generate some values
                                    else: # greater than
                                        distance_from_mean = columnvec.max() - columnvec.mean()
                                        delta = (distance_from_mean * 3.1 / 3) - distance_from_mean # we would like a bell curve where the third st.dev reaches 3.1 times the distance from mean
                                        dist = (micro_diff + np.random.gamma(shape=1, scale=delta, size=size))[:, np.newaxis] # generate some values
                        else: # nominal
                            print('failed to find/synthesise cover on a nominal feature')
                            # fails to create dist - should error until fixed
                rule_covered_dists = deepcopy(instances)
                for j, f in enumerate(features):
                    if f == feature: # binary encoded feature
                        rule_covered_dists[:, j] = dist
                # end of inner except
            mask_cover = False
            # end of outer except

        # create a matrix of identical instances, non-sparse to optimise columnwise ops
        mask_matrix = np.repeat(instance.todense(), size, axis=0)
        # for instance specific mask
        # we want the feature that was changed in the rule complement to be unmasked
        # beware of binary encoded features
        if var_dict[feature]['data_type'] == 'continuous':
            to_unmask = [feature]
        else:
            to_unmask = var_dict[feature]['labels_enc']

        mask_matrix_is = deepcopy(mask_matrix)
        # this will update the instance mask with the reversed feature
        for j, f in enumerate(features): # by column
            if f in to_unmask: # binary encoded feature
                mask_matrix_is[:, j] = rule_covered_dists[:, j].todense()

        # prepare additional set of columns to unmask
        parent_features = self.categorise_rule_features(rule=rule,
                                                        var_dict=var_dict,
                                                        var_dict_enc=var_dict_enc)

        if feature in parent_features.keys(): # it won't be there in the case of a non-covered rule
            del parent_features[feature]

        to_unmask = []
        for prnt in parent_features:
            if var_dict[prnt]['data_type'] == 'continuous':
                to_unmask.append(prnt)
            else:
                to_unmask = to_unmask + var_dict[prnt]['labels_enc'] # simple concatenation to avoid nesting

        # copy the is output and introduce other allowed values
        mask_matrix_av = deepcopy(mask_matrix_is)
        for j, f in enumerate(features): # by column
            if f in to_unmask: # binary encoded feature
                mask_matrix_av[:, j] = rule_covered_dists[:, j].todense()

        return(mask_matrix_is, mask_matrix_av, mask_cover)

    def get_alt_labelings(self, forest, instance, sample_instances,
                            rule_complements=None,
                            var_dict=None,
                            sample_labels=None):
        # general setup
        instances, _ = self.init_instances(instances=sample_instances)
        if rule_complements is None:
            rule_complements = self.get_rule_complements()
        size = instances.shape[0]
        alt_labelings_results = []
        # for each rule comp, create datasets of the same size as the leave-one-out test set
        for feature in rule_complements:
            rc = rule_complements[feature]
            instance_specific_mask, allowed_values_mask, mask_cover = self.mask_by_instance(instance=instance,
                                                                                                            sample_instances=instances,
                                                                                                            rule=rc, feature=feature,
                                                                                                            size=size)
            ism_preds = forest.predict(instance_specific_mask)
            ism_post = p_count_corrected(ism_preds, [i for i in range(len(self.class_names))])

            avm_preds = forest.predict(allowed_values_mask)
            avm_post = p_count_corrected(avm_preds, [i for i in range(len(self.class_names))])

            alt_labelings_results.append({'feature' : feature,
                                            'is_mask' : ism_post,
                                            'av_mask' : avm_post,
                                            'mask_cover' : mask_cover})

        return(alt_labelings_results)

    def to_screen(self):
        print('Model Results for Instance')
        print('target (predicted) class: ' + str(self.target_class) + ' (' + self.target_class_label + ')')
        print('target class prior (training data): ' + str(self.prior[self.target_class]))
        print('forest vote share (unseen instance): ' + str(self.forest_vote_share))
        print('forest vote margin (unseen instance): ' + str(self.forest_vote_margin))
        print('confidence weighted forest vote share (unseen instance): ' + str(self.conf_weight_forest_vote_share))
        print('confidence weighted forest vote margin (unseen instance): ' + str(self.conf_weight_forest_vote_margin))
        print()
        print('rule: ' + self.pretty_rule)
        print('rule cardinality: ' + str(self.rule_len))
        print('accumulated weight of rule: ' + str(self.accumulated_weights))
        print()
        print('Estimated Results - Rule Training Sample. Algorithm: ' + self.algorithm)
        print('rule coverage (training data): ' + str(self.est_coverage))
        print('rule xcoverage (training data): ' + str(self.est_xcoverage))
        print('rule precision (training data): ' + str(self.est_prec))
        print('rule stability (training data): ' + str(self.est_stab))
        print('rule recall (training data): ' + str(self.est_recall))
        print('rule f1 score (training data): ' + str(self.est_f1))
        print('rule NPV (training data): ' + str(self.est_npv))
        print('rule lift (training data): ' + str(self.est_lift))
        print('prior (training data): ' + str(self.prior))
        print('prior counts (training data): ' + str(self.prior_counts))
        print('rule posterior (training data): ' + str(list(reversed(self.posterior))[0]))
        print('rule posterior counts (training data): ' + str(self.posterior_counts))
        print('rule chisq p-value (training data): ' + str(chisq_indep_test(self.posterior_counts, self.prior_counts)[1]))
        print('rule Kullback-Leibler divergence (training data): ' + str(self.est_kl_div))
        print()

    def to_dict(self):
        return({'features' : self.features,
        'features_enc' : self.features_enc,
        'class_names' : self.class_names,
        'var_dict' : self.var_dict,
        'var_dict_enc' : self.var_dict_enc,
        'paths' : self.paths,
        'patterns' : self.patterns,
        'rule' : self.rule,
        'pruned_rule' : self.pruned_rule,
        'target_class' :self.target_class,
        'target_class_label' :self.target_class_label,
        'major_class' : self.major_class,
        'major_class_label' :self.major_class_label,
        'model_votes' : self.model_votes,
        'confidence_weights' : self.confidence_weights,
        'accumulated_weights' : self.accumulated_weights,
        'posterior' : self.posterior,
        'stability' : self.stability,
        'accuracy' : self.accuracy,
        'counts' : self.counts,
        'recall' : self.recall,
        'f1' : self.f1,
        'cc' : self.cc,
        'ci' : self.ci,
        'ncc' : self.ncc,
        'nci' : self.nci,
        'npv' : self.npv,
        'lift' : self.lift,
        'chisq' : self.chisq,
        'algorithm' : algorithm})

# this class runs all steps of the CHIRPS algorithm
class CHIRPS_runner(rule_evaluator):

    def __init__(self, meta_data,
                paths, paths_weights,
                paths_pred_proba,
                tree_preds,
                model_votes,
                confidence_weights,
                major_class=None,
                target_class=None,
                random_state=123,
                patterns=None):

        self.random_state = random_state
        self.paths = paths
        self.paths_weights = paths_weights
        self.paths_pred_proba = paths_pred_proba
        self.tree_preds = tree_preds
        self.model_votes = model_votes
        self.confidence_weights = confidence_weights
        self.major_class = major_class
        self.target_class = target_class
        self.patterns = patterns

        self.features = meta_data['features']
        self.features_enc = meta_data['features_enc']
        self.var_dict = meta_data['var_dict']
        self.var_dict_enc = meta_data['var_dict_enc']
        self.class_col = meta_data['class_col']
        self.n_classes = len(meta_data['class_names'])

        meta_le_dict = meta_data['le_dict']
        meta_get_label = meta_data['get_label']
        meta_class_names = meta_data['class_names']
        if self.class_col in meta_le_dict.keys():
            self.get_label = meta_get_label
            self.class_names = self.get_label(self.class_col, [i for i in range(len(meta_class_names))])
        else:
            self.get_label = None
            self.class_names = meta_class_names

        for item in self.var_dict:
            if self.var_dict[item]['class_col']:
                continue
            else:
                if self.var_dict[item]['data_type'] == 'nominal':
                    n_labs = len(self.var_dict[item]['labels'])
                else:
                    n_labs = 1
                self.var_dict[item]['upper_bound'] = [math.inf] * n_labs
                self.var_dict[item]['lower_bound'] = [-math.inf] * n_labs
        self.rule = []
        self.pruned_rule = []
        self.__previous_rule = []
        self.__reverted = []
        self.total_points = None
        self.accumulated_points = 0
        self.accumulated_weights = 0
        self.sample_instances = None
        self.sample_labels = None
        self.n_instances = None
        self.target_class = None
        self.target_class_label = None
        self.major_class = None
        self.prior_info = None
        self.posterior = None
        self.stability = None
        self.accuracy = None
        self.counts = None
        self.recall = None
        self.f1 = None
        self.cc = None
        self.ci = None
        self.ncc = None
        self.nci = None
        self.npv = None
        self.lift = None
        self.chisq = []
        self.kl_div = []
        self.isolation_pos = None
        self.stopping_param = None
        self.merge_rule_iter = None
        self.algorithm = None

    def discretize_paths(self, bins=4, equal_counts=False, var_dict=None):
        # check if bins is not numeric or can't be cast, then force equal width (equal_counts = False)
        var_dict, _ = self.init_dicts(var_dict=var_dict)

        if equal_counts:
            def hist_func(x, bins, weights=None):
                npt = len(x)
                bns = np.interp(np.linspace(0, npt, bins + 1),
                                 np.arange(npt),
                                 np.sort(x))
                return(np.histogram(x, bns, weights=weights))
        else:
            def hist_func(x, bins, weights=None):
                return(np.histogram(x, bins, weights=weights))

        cont_vars = [vn for vn in var_dict if var_dict[vn]['data_type'] == 'continuous' and var_dict[vn]['class_col'] == False]
        for feature in cont_vars:

            # lower bound, greater than
            lowers = [item[2] for nodes in self.paths for item in nodes if item[0] == feature and item[1] == False]

            # upper bound, less than
            uppers = [item[2] for nodes in self.paths for item in nodes if item[0] == feature and item[1] == True]

            if uppers:
                upper_bins = hist_func(uppers, bins=bins)[1]
            else:
                upper_bins = np.repeat(np.nan, bins)

            if lowers:
                lower_bins = hist_func(lowers, bins=bins)[1]
            else:
                lower_bins = np.repeat(np.nan, bins)

            upper_bin_midpoints = Series(upper_bins).rolling(window=2, center=False).mean().values[1:]
            upper_bin_means = (np.histogram(uppers, upper_bins, weights=uppers)[0] /
                                np.histogram(uppers, upper_bins)[0]).round(5) # can result in nans if no value falls into bin
            upper_bin_mids = [i if not np.isnan(i) else j for i, j in zip(upper_bin_means, upper_bin_midpoints)]

            lower_bin_midpoints = Series(lower_bins).rolling(window=2, center=False).mean().values[1:]
            lower_bin_means = (np.histogram(lowers, lower_bins, weights=lowers)[0] /
                                np.histogram(lowers, lower_bins)[0]).round(5) # can result in nans
            lower_bin_mids = [i if not np.isnan(i) else j for i, j in zip(lower_bin_means, lower_bin_midpoints)]

                        # discretize functions from histogram means
            upper_discretize = lambda x: upper_bin_mids[np.max([np.min([np.digitize(x, upper_bins), len(upper_bin_mids)]), 1]) - 1]
            lower_discretize = lambda x: lower_bin_mids[np.max([np.min([np.digitize(x, lower_bins, right= True), len(upper_bin_mids)]), 1]) - 1]

            paths_discretized = []
            for nodes in self.paths:
                nodes_discretized = []
                for f, t, v in nodes:
                    if f == feature:
                        if t == False: # greater than, lower bound
                            v = lower_discretize(v)
                        else:
                            v = upper_discretize(v)
                    nodes_discretized.append((f, t, v))
                paths_discretized.append(nodes_discretized)
            # at the end of each loop, update the instance variable
            self.paths = paths_discretized

    def mine_patterns(self, paths_lengths_threshold=2, support=0.1):

        # repeat paths if max length > path length threshold
        # e.g. for boosted models with stumps of depth 1 or 2, it doesn't make much sense
        # for longer paths, the boosting weight is used to increase the support count
        if len(max(self.paths, key=len)) >= paths_lengths_threshold:

            # ensure support to an absolute number of instances rather than a fraction
            if support <= 1:
                support = round(support * len(self.paths))

            # normalise the weights so min(weights) = 1.0
            weighted_counts = np.round(self.paths_weights * 1/min(self.paths_weights)).astype('int')

            # replicate the paths a number of times according to weighted counts
            self.paths = list(chain.from_iterable(map(repeat, self.paths, weighted_counts)))

            # FP mining
            self.patterns = find_frequent_patterns(self.paths, support)
            # normalise support score
            self.patterns = {patt : self.patterns[patt]/len(self.paths) for patt in self.patterns}

        # otherwise, convert paths to patterns giving weights as support
        else:
            # ensure support to a fraction
            if support > 1:
                support = support / len(self.paths)

            entropy_weighted_patterns = defaultdict(np.float32)
            instances, labels = self.init_instances()
            prior = p_count_corrected(labels, [i for i in range(len(self.class_names))])
            prior_counts = prior['counts']

            # neutral estimator weights - not AdaBoost
            if np.all([i == 1.0 for i in self.paths_weights]):
                # weight by how well it discriminates - how different from prior, based on kl-div
                paths_weights = [contingency_test(ppp, prior['p_counts'], 'kldiv') for ppp in self.paths_pred_proba]
            else:
                # otherwise the weights from AdaBoost
                paths_weights = self.paths_weights

            for j, p in enumerate(self.paths):
                items = []
                kldivs = []
                for item in p:
                    # [item] because it's length is one and therefore would iterates into a character list
                    idx = self.apply_rule(rule=[item], instances=instances, features=self.features_enc)
                    p_counts = p_count_corrected(labels[idx], [i for i in range(len(self.class_names))])
                    covered = p_counts['counts']
                    # collect the entropy for each node in the tree/stump: how well does individual node discriminate?
                    kldivs.append(contingency_test(covered, prior_counts, 'kldiv'))
                    items.append(item)
                for e, item in zip(kldivs, items):
                    # running sum of the normalised then tree-weighted entropy for any node found in the ensemble
                    entropy_weighted_patterns[item] += e / sum(kldivs) * paths_weights[j]
            # normalise the partial weighted entropy so it can be filtered by suppert (support takes on a slightly different meaning here)
            self.patterns = {((p), ) : w / sum(dict(entropy_weighted_patterns).values()) \
                                        for p, w in dict(entropy_weighted_patterns).items() \
                                        if w / sum(dict(entropy_weighted_patterns).values()) >= support }


    def mine_path_snippets(self, paths_lengths_threshold=2, support_paths=0.1,
                            disc_path_bins=4, disc_path_eqcounts=False):

        # discretize any numeric features
        self.discretize_paths(bins=disc_path_bins,
                                equal_counts=disc_path_eqcounts)

        # the patterns are found but not scored and sorted yet
        self.mine_patterns(paths_lengths_threshold=paths_lengths_threshold, support=support_paths)

    def sort_patterns(self, alpha=0.0, weights=None, score_func=1):
        alpha = float(alpha)
        if weights is None:
            weights = [1] * len(self.patterns)

        # to shrink the support of shorter freq_patterns
        # formula is sqrt(weight) * sup * ()(len - alpha) / len)
        if score_func == 1:
            score_function = lambda x, w: (x[0], x[1], w * x[1] * (len(x[0]) - alpha) / len(x[0]))
        # alternatives
        elif score_func == 2:
            score_function = lambda x, w: (x[0], x[1], w * x[1] * (len(x[0]) - alpha) / (len(x[0])**2))
        elif score_func == 3:
            score_function = lambda x, w: (x[0], x[1], w * (len(x[0]) - alpha) / len(x[0]))
        elif score_func == 4:
            score_function = lambda x, w: (x[0], x[1], w * (len(x[0]) - alpha) / (len(x[0])**2))
        else: # weights only
            score_function = lambda x, w: (x[0], x[1], w)
        fp_scope = [fp for fp in map(score_function, self.patterns.items(), weights)]
        # score is now at position 2 of tuple
        self.patterns = sorted(fp_scope, key=itemgetter(2), reverse = True)

    def score_sort_path_snippets(self, sample_instances, sample_labels, target_class=None,
                                    alpha_paths=0.0, score_func=1, weighting='chisq'):
        # best at -1 < alpha < 1
        # now the patterns are scored and sorted. alpha > 0 favours longer patterns. 0 neutral. < 0 shorter.
        # the patterns can be weighted by chi**2 for independence test, p-values # TODO KL-div, log odds
        if weighting is None:
            self.sort_patterns(alpha=alpha_paths, score_func=score_func) # with only support/alpha sorting
        else:
            weights = [1] * len(self.patterns) # default neutral if no valid combo
            for j, wp in enumerate(self.patterns):

                instances, labels = self.init_instances(instances=sample_instances, labels=sample_labels)
                idx = self.apply_rule(rule=wp, instances=instances, features=self.features_enc)
                p_counts_covered = p_count_corrected(sample_labels[idx], [i for i in range(len(self.class_names))])
                covered = p_counts_covered['counts']
                not_covered = p_count_corrected(labels[~idx], [i for i in range(len(self.class_names))])['counts']
                all_instances = p_count_corrected(labels, [i for i in range(len(self.class_names))])['counts']

                # observed = np.array((covered, not_covered))
                observed = np.array((covered, all_instances))

                if weighting in ['chisq', 'kldiv', 'lodds']:
                    weights[j] = contingency_test(covered, all_instances, weighting)

                # rf+hc score or no
                else: #
                    if target_class is not None:
                        cc = p_counts_covered['counts'][target_class]
                        ic = sum([c for c, l in zip(p_counts_covered['counts'], p_counts_covered['labels']) if l != target_class])
                        if weighting == 'rf+hc1':
                            weights[j] = (cc-ic) / (cc+ic) + cc / (ic + 1) # 4 is given in the paper. it can be anything to prevent div by zero
                        elif weighting == 'rf+hc2':
                            weights[j] = (cc-ic) / (cc+ic) + cc / (ic + 1) + cc / len(wp[0])
                        else:
                            pass
                    else:
                        pass

            # correct any uncalculable weights
            weights = [w if not n else min(weights) for w, n in zip(weights, np.isnan(weights))] # clean up any nans
            # shift to zero or above
            weights = [w + abs(min(weights)) for w in weights]
            # normalise
            weights = [w/max(weights) for w in weights]
            # final application of weights
            self.sort_patterns(alpha=alpha_paths, score_func=score_func, weights=weights)

    def add_rule_term(self):
        self.__previous_rule = deepcopy(self.rule)
        next_rule = self.patterns[self.unapplied_rules[0]]
        candidate = [] # to be output and can be rejected and reverted if no improvement to target function
        for item in next_rule[0]:
            # list of already used features
            # to be created each item iteration
            # as the order is important can be rarranged by inserts
            feature_appears = [f for (f, _, _) in self.rule]

            # skip duplicates (essential for pruning reasons)
            if item in self.rule:
                continue

            if item[0] in self.var_dict_enc: # binary feature
                # find the parent feature of item
                parent_feature = self.var_dict_enc[item[0]]

                # check for any known True feature value
                if any(np.array(self.var_dict[parent_feature]['lower_bound']) > 0):
                    continue

                # list of already used categorical parent features
                # to be created each item iteration
                # as the order is important can be rarranged by inserts
                categorical_feature_appears = []
                for f_app in feature_appears:
                    if f_app in self.var_dict_enc.keys(): # it is an encoded categorical
                        categorical_feature_appears.append(self.var_dict_enc[f_app])
                    else: # it is continuous
                        categorical_feature_appears.append(f_app)
                # insert item after last position in current rule where parent item appears
                if parent_feature in categorical_feature_appears:
                    self.rule.insert(max(np.where(np.array(categorical_feature_appears) == parent_feature)[0]) + 1, item)
                # otherwise just append to current rule
                else:
                    self.rule.append(item)
                candidate.append(item) # this will output the newly added terms

            else: # continuous feature
                append_or_update = False
                if item[1]: # leq_threshold True
                    if item[2] <= self.var_dict[item[0]]['upper_bound'][0]:
                        append_or_update = True

                else:
                    if item[2] > self.var_dict[item[0]]['lower_bound'][0]:
                        append_or_update = True

                if append_or_update:
                    if item[0] in feature_appears:
                        # print(item, 'feature appears already')
                        valueless_rule = [(f, t) for (f, t, _) in self.rule]
                        if (item[0], item[1]) in valueless_rule: # it's already there and needs updating
                            # print(item, 'feature values appears already')
                            self.rule[valueless_rule.index((item[0], item[1]))] = item
                        else: # feature has been used at the opposite end (either lower or upper bound) and needs inserting
                            # print(item, 'feature values with new discontinuity')
                            self.rule.insert(feature_appears.index(item[0]) + 1, item)
                    else:
                        # print(item, 'feature first added')
                        self.rule.append(item)
                    candidate.append(item) # this will output the newly added terms

        # accumlate points from rule and tidy up
        # remove the first item from unapplied_rules as it's just been applied or ignored for being out of range
        self.accumulated_points += self.patterns[self.unapplied_rules[0]][2]
        self.accumulated_weights += self.patterns[self.unapplied_rules[0]][1]
        del self.unapplied_rules[0]
        # accumlate all the freq patts that are subsets of the current rules
        # remove the index from the unapplied rules list (including the current rule just added)
        to_remove = []
        for ur in self.unapplied_rules:
            # check if all items are already part of the rule (i.e. it's a subset)
            if all([item in self.rule for item in self.patterns[ur][0]]):
                self.accumulated_points += self.patterns[ur][2]
                self.accumulated_weights += self.patterns[ur][1]
                # collect up the values to remove. don't want to edit the iterator in progress
                to_remove.append(ur)
        for rmv in reversed(to_remove):
            self.unapplied_rules.remove(rmv)

        return(candidate)

    def prune_rule(self):
        # removes all other binary items if one Greater than is found.

        # find any nominal binary encoded feature value and its parent if appears as False (greater than)
        gt_items = {}
        for item in self.rule:
            if not item[1] and item[0] in self.var_dict_enc: # item is greater than thresh (False valued) and a nominal type
                gt_items.update({ self.var_dict_enc[item[0]] : item[0] }) # capture the parent feature and the feature value / there can only be one true

        gt_pruned_rule = [] # captures binary encoded variables
        for item in self.rule:
            if item[0] in self.var_dict_enc:
                if self.var_dict_enc[item[0]] not in gt_items.keys(): # item parent not in the thresh False set captured just above
                    gt_pruned_rule.append(item)
                elif not item[1]: # any item thresh False valued (it will be in the thresh False set above)
                    gt_pruned_rule.append(item)
            else: # continuous
                gt_pruned_rule.append(item)

        # if all but one of a feature set is False, swap them out for the remaining value
        # start by counting all the lt thresholds in each parent feature
        lt_items = defaultdict(lambda: 0)
        for item in gt_pruned_rule:
            if item[1] and item[0] in self.var_dict_enc: # item is less than thresh (True valued) and a nominal type
                lt_items[self.var_dict_enc[item[0]]] += 1 # capture the parent feature and count each True valued feature value

        # checking if just one other feature value remains unused
        pruned_items = [item[0] for item in gt_pruned_rule]
        for lt in dict(lt_items).keys(): # convert from defaultdict to dict for counting keys
            n_categories = len([i for i in self.var_dict_enc.values() if i == lt])
            if n_categories - dict(lt_items)[lt] == 1:
                # get the remaining value for this feature
                lt_labels = self.var_dict[lt]['labels_enc']
                to_remove = [label for label in lt_labels if label in pruned_items]
                remaining_value = [label for label in lt_labels if label not in pruned_items]

                # update the feature dict as the one true result might not have been seen
                pos = self.var_dict[lt]['labels_enc'].index(remaining_value[0])
                self.var_dict[lt]['lower_bound'][pos] = 0.5

                # this is to scan the rule and put feature values with the same parent side by side
                lt_pruned_rule = []
                pos = -1
                for rule in gt_pruned_rule:
                    pos += 1
                    if rule[0] not in to_remove:
                        lt_pruned_rule.append(rule)
                    else:
                        # set the position of the last term of the parent feature
                        insert_pos = pos
                        pos -= 1
                lt_pruned_rule.insert(insert_pos, (remaining_value[0], False, 0.5))

                # the main rule is updated for passing through the loop again
                gt_pruned_rule = lt_pruned_rule.copy()

        self.pruned_rule = gt_pruned_rule

    def __greedy_commit__(self, value, threshold):
        if value <= threshold:
            self.rule = deepcopy(self.__previous_rule)
            self.__reverted.append(True)
            return(True)
        else:
            self.__reverted.append(False)
            return(False)

    def merge_rule(self, sample_instances, sample_labels, forest,
                        stopping_param = 1,
                        precis_threshold = 0.95,
                        fixed_length = None,
                        target_class = None,
                        algorithm='greedy_stab',
                        merging_bootstraps = 0,
                        pruning_bootstraps = 0,
                        bootstrap_confidence = 0.95,
                        delta = 0.1,
                        random_state=None):

        instances, labels = self.init_instances(instances=sample_instances, labels=sample_labels)
        self.unapplied_rules = [i for i in range(len(self.patterns))]
        default_rule = []

        if len(self.unapplied_rules) == 0:
            self.total_points = 0
        else:
            default_rule = self.patterns[0][0]
            # default_rule will be set in the loop
            self.total_points = sum([scrs[2] for scrs in self.patterns])

        # basic setup
        # pointless to receive a None for algorithm
        if algorithm is None:
            self.algorithm = 'greedy_stab'
        else:
            self.algorithm = algorithm
        # common default setting: see class non_deterministic
        random_state = self.default_if_none_random_state(random_state)
        np.random.seed(random_state) # for bootstrap pruning
        if stopping_param > 1 or stopping_param < 0:
            stopping_param = 1
            print('warning: stopping_param should be 0 <= p <= 1. Value was reset to 1')
        self.stopping_param = stopping_param
        self.n_classes = len(np.unique(labels))
        self.n_instances = len(labels)

        # model predicted class
        if self.major_class is None:
            self.major_class = np.argmax(self.model_votes['p_counts'])
        if self.get_label is None:
            self.major_class_label = self.major_class
        else:
            self.major_class_label = self.get_label(self.class_col, [self.major_class])

        # target class
        if target_class is None and self.target_class is None:
            self.target_class = self.major_class
            self.target_class_label = self.major_class_label
        elif target_class is not None:
            self.target_class = target_class
        if self.get_label is None:
            self.target_class_label = self.target_class
        else:
            self.target_class_label = self.get_label(self.class_col, [self.target_class])

        # prior - empty rule
        prior_eval = self.evaluate(labels, np.full(self.n_instances, True))
        self.posterior = np.array([prior_eval['posterior'].tolist()])
        self.counts = np.array([prior_eval['counts'].tolist()])

        self.stability = np.array([prior_eval['stability'].tolist()])

        self.recall = [np.full(self.n_classes, 1.0)] # counts / prior counts
        self.f1 =  [2] * ( ( self.posterior * self.recall ) / ( self.posterior + self.recall ) ) # 2 * (precis * recall/(precis + recall) )
        self.accuracy = np.array([prior_eval['accuracy'].tolist()])
        self.lift = [np.full(self.n_classes, 1.0)] # precis / (total_cover * prior)
        self.cc = np.array([prior_eval['cc'].tolist()])
        self.ci = np.array([prior_eval['ci'].tolist()])
        self.ncc = np.array([prior_eval['ncc'].tolist()])
        self.nci = np.array([prior_eval['nci'].tolist()])
        self.npv = np.array([prior_eval['npv'].tolist()])

        # pre-loop set up
        # rule based measures - prior/empty rule
        current_precision = prior_eval['posterior'][np.where(prior_eval['labels'] == self.target_class)][0] # based on prior

        # accumulate rule terms
        rule_length_counter = 0
        self.merge_rule_iter = 0
        default_metric = 0.0

        while current_precision != 1.0 \
            and current_precision != 0.0 \
            and current_precision < precis_threshold \
            and self.accumulated_points <= self.total_points * self.stopping_param \
            and (fixed_length is None or rule_length_counter < max(1, fixed_length)) \
            and len(self.unapplied_rules) > 0:

            self.merge_rule_iter += 1
            candidate = self.add_rule_term()
            eval_rule = self.evaluate_rule(sample_instances=instances,
                                    sample_labels=labels)
            # confirm rule, or revert to previous
            # choosing from a range of possible metrics and learning improvement
            # e.g if there was no change, or a decrease then reject, roll back and take the next one
            if self.algorithm == 'greedy_prec':
                metric = 'posterior'
                prev = self.posterior
            elif self.algorithm == 'greedy_f1':
                metric = 'f1'
                prev = self.f1
            elif self.algorithm == 'greedy_acc':
                metric = 'accuracy'
                prev = self.accuracy
            else: # 'greedy_stab'
                metric = 'stability'
                prev = self.stability

            curr = eval_rule[metric]
            current = curr[np.where(eval_rule['labels'] == self.target_class)]
            previous = list(reversed(prev))[0][np.where(eval_rule['labels'] == self.target_class)]

            if rule_length_counter == 0 and current > default_metric: # we need a default rule
                default_rule = self.rule # whatever was just set as the candidate
                default_metric = current[0]

            if merging_bootstraps == 0:
                should_continue = self.__greedy_commit__(current, previous)
            else: # get a bootstrapped evaluation
                b_curr = np.full(merging_bootstraps, np.nan)
                b_prev = np.full(merging_bootstraps, np.nan)
                for b in range(merging_bootstraps):

                    idx = np.random.choice(self.n_instances, size = self.n_instances, replace=True)

                    b_sample_instances = instances[idx]
                    b_sample_labels = labels[idx]

                    b_eval_rule = self.evaluate_rule(sample_instances=b_sample_instances,
                                                sample_labels=b_sample_labels)
                    b_curr[b] = b_eval_rule[metric][np.where(eval_rule['labels'] == self.target_class)]

                    b_eval_prev = self.evaluate_rule(rule = self.__previous_rule,
                                                sample_instances=b_sample_instances,
                                                sample_labels=b_sample_labels)
                    b_prev[b] = b_eval_prev[metric][np.where(b_eval_prev['labels'] == self.target_class)]

                should_continue = self.__greedy_commit__((b_curr > b_prev).sum(), bootstrap_confidence * merging_bootstraps)

            if should_continue:
                continue # don't update all the metrics, just go to the next round
            # otherwise update everything and save all the metrics
            rule_length_counter += 1

            # check for end conditions; no target class instances
            if eval_rule['posterior'][np.where(eval_rule['labels'] == self.target_class)] == 0.0:
                current_precision = 0.0
            else:
                current_precision = eval_rule['posterior'][np.where(eval_rule['labels'] == self.target_class)][0]

            # per class measures
            self.posterior = np.append(self.posterior, [eval_rule['posterior']], axis=0)
            self.stability = np.append(self.stability, [eval_rule['stability']], axis=0)
            self.counts = np.append(self.counts, [eval_rule['counts']], axis=0)
            self.accuracy = np.append(self.accuracy, [eval_rule['accuracy']], axis=0)
            self.recall = np.append(self.recall, [eval_rule['recall']], axis=0 )
            self.f1 = np.append(self.f1, [eval_rule['f1']], axis=0 )
            self.lift = np.append(self.lift, [eval_rule['lift']], axis=0 )
            self.chisq = np.append(self.chisq, [eval_rule['chisq']], axis=0 ) # p-value
            self.kl_div = np.append(self.kl_div, [eval_rule['kl_div']], axis=0 )
            self.cc = np.append(self.cc, [eval_rule['cc']], axis=0 )
            self.ci = np.append(self.ci, [eval_rule['ci']], axis=0 )
            self.ncc = np.append(self.ncc, [eval_rule['ncc']], axis=0 )
            self.nci = np.append(self.nci, [eval_rule['nci']], axis=0 )
            self.npv = np.append(self.npv, [eval_rule['npv']], axis=0 )

            # update the var_dict with the rule values
            for item in candidate:
                if item[0] in self.var_dict_enc: # binary feature
                    # find the parent feature of item
                    parent_feature = self.var_dict_enc[item[0]]

                    # update the var_dict
                    position = self.var_dict[parent_feature]['labels_enc'].index(item[0])
                    if item[1]: # leq_threshold True
                        self.var_dict[parent_feature]['upper_bound'][position] = item[2]
                    else: # a known True feature value
                        self.var_dict[parent_feature]['lower_bound'][position] = item[2]
                        # set all other options to False (less that 0.5 is True i.e. False = 0)
                        ub = [item[2]] * len(self.var_dict[parent_feature]['upper_bound'])
                        ub[position] = np.inf
                        self.var_dict[parent_feature]['upper_bound'] = ub

                else: # continuous
                    if item[1]: # leq_threshold True
                        if item[2] <= self.var_dict[item[0]]['upper_bound'][0]:
                            self.var_dict[item[0]]['upper_bound'][0] = item[2]

                    else:
                        if item[2] > self.var_dict[item[0]]['lower_bound'][0]:
                            self.var_dict[item[0]]['lower_bound'][0] = item[2]

        # case no solution was found
        if rule_length_counter == 0:
            print("no solution")
            self.rule = default_rule
        if len(self.kl_div) == 0:
            self.kl_div = np.append(self.kl_div, [0], axis=0 )

        # first time major_class is isolated
        if any(np.argmax(self.posterior, axis=1) == self.target_class):
            self.isolation_pos = np.min(np.where(np.argmax(self.posterior, axis=1) == self.target_class))
        else: self.isolation_pos = None

        # set up the rule for clean up
        self.prune_rule()

        # rule complement testing to remove any redundant rule terms

        if pruning_bootstraps > 0:
            curr_len = len(self.pruned_rule) + 1
            new_len = curr_len - 1
            current_rule = self.pruned_rule
            while curr_len > new_len and new_len > 1: # repeat until no more improvement, stop before removing last rule
                # get a bootstrapped evaluation
                b_current = np.full(pruning_bootstraps, np.nan)
                for b in range(pruning_bootstraps):
                    idx = np.random.choice(self.n_instances, size = self.n_instances, replace=True)
                    b_sample_instances = instances[idx]
                    b_sample_labels = labels[idx]

                    eval_current = self.evaluate_rule(rule=current_rule, sample_instances=b_sample_instances, sample_labels=b_sample_labels)
                    eval_current_post = eval_current['posterior']
                    eval_current_counts = eval_current['counts']
                    b_current[b] = eval_current_post[np.where(eval_rule['labels'] == self.target_class)]

                    rule_complement_results = self.eval_rule_complements(sample_instances=b_sample_instances, sample_labels=b_sample_labels)
                    n_rule_complements = len(rule_complement_results)
                    b_rc = np.full(n_rule_complements, np.nan)

                    for rc, rcr in enumerate(rule_complement_results):
                        eval_rcr = rcr['eval']
                        rcr_posterior = eval_rcr['posterior']
                        b_rc[rc] = rcr_posterior[np.where(eval_rcr['labels'] == self.target_class)]
                        # rcr_posterior_counts = eval_rcr['counts']

                    if b == 0:
                        b_rcr = np.array(b_rc)
                    else:
                        b_rcr = np.vstack((b_rcr, b_rc))

                to_remove = []
                for rc in range(n_rule_complements):
                    if (b_current - delta >= b_rcr[:,rc]).sum() < bootstrap_confidence * pruning_bootstraps:
                            if self.var_dict[rule_complement_results[rc]['feature']]['data_type'] == 'nominal':
                                to_remove = to_remove + self.var_dict[rule_complement_results[rc]['feature']]['labels_enc']
                            else:
                                to_remove = to_remove + [rule_complement_results[rc]['feature']]

                self.__previous_rule = self.pruned_rule
                self.pruned_rule = [(f, t, v) for f, t, v in self.pruned_rule if f not in to_remove]
                curr_len = new_len
                new_len = len(self.pruned_rule)
        # else: do nothing

        # belt and braces - if rule pruning drops everything
        if len(self.pruned_rule) == 0:
            print("solution pruned away. defaulting")
            self.pruned_rule = [self.__previous_rule[np.argmin(np.mean(b_rcr, axis=0))]] # keep the one that drops precision the most

            # self.rule = self.pruned_rule[0]
            # print("solution pruned away. defaulting")
            # print(default_rule)
            # self.rule = default_rule
            # self.pruned_rule = default_rule


    def get_CHIRPS_explainer(self, elapsed_time=0):

        return(CHIRPS_explainer(self.random_state,
        self.features, self.features_enc, self.class_names,
        self.class_col, self.get_label,
        self.var_dict, self.var_dict_enc,
        self.paths, self.paths_weights, self.patterns,
        self.rule, self.pruned_rule,
        self.target_class, self.target_class_label,
        self.major_class, self.major_class_label,
        self.model_votes,
        self.confidence_weights,
        self.accumulated_weights,
        self.isolation_pos,
        self.posterior,
        self.stability,
        self.accuracy,
        self.counts,
        self.recall,
        self.f1,
        self.cc,
        self.ci,
        self.ncc,
        self.nci,
        self.npv,
        self.lift,
        self.chisq,
        self.kl_div,
        self.algorithm,
        elapsed_time))

class batch_CHIRPS_explainer(object):

    def __init__(self, bp_container, # batch_paths_container
                        forest, sample_instances, sample_labels, meta_data,
                        forest_walk_mean_elapsed_time=0):
        self.bp_container = bp_container
        self.data_container = data_container
        self.forest = forest
        self.sample_instances = sample_instances
        self.sample_labels = sample_labels
        self.meta_data = meta_data
        self.CHIRPS_explainers = None
        self.fwmet = forest_walk_mean_elapsed_time

    def batch_run_CHIRPS(self, target_classes=None,
                        chirps_explanation_async=False,
                        random_state=123,
                        **kwargs):
        # defaults
        options = {
            'paths_lengths_threshold' : 2,
            'support_paths' : 0.05,
            'alpha_paths' : 0.0,
            'disc_path_bins' : 4,
            'disc_path_eqcounts' : False,
            'score_func' : 1,
            'which_trees' : 'majority',
            'precis_threshold' : 0.95,
            'weighting' : 'chisq',
            'algorithm' : 'greedy_stab',
            'merging_bootstraps' : 20,
            'pruning_bootstraps' : 20,
            'delta' : 0.1 }
        options.update(kwargs)

        # convenience function to orient the top level of bpc
        # a bit like reshaping an array
        # reason: rf paths quickly extracted per tree for all instances
        # so when constructed, this structure is oriented by tree
        # and we would like to easily iterate by instance
        if self.bp_container.by_tree:
            self.bp_container.flip()
        n_instances = len(self.bp_container.path_detail)

        if target_classes is None:
            target_classes = [None] * n_instances
        # initialise a list for the results
        CHIRPS_explainers = [[]] * n_instances
        # generate the explanations
        if chirps_explanation_async:

            async_out = []
            n_cores = mp.cpu_count()-2
            pool = mp.Pool(processes=n_cores)

            # loop for each instance
            for i in range(n_instances):
                # get a CHIRPS_runner per instance
                # filtering by the chosen set of trees - default: majority voting
                # use deepcopy to ensure by_value, not by_reference instantiation
                c_runner = self.bp_container.get_CHIRPS_runner(i, deepcopy(self.meta_data), which_trees=options['which_trees'], random_state=random_state)
                # run the chirps process on each instance paths
                async_out.append(pool.apply_async(as_CHIRPS,
                    (c_runner, target_classes[i],
                    self.sample_instances, self.sample_labels,
                    self.forest, self.fwmet,
                    options['paths_lengths_threshold'], options['support_paths'], options['alpha_paths'],
                    options['disc_path_bins'], options['disc_path_eqcounts'], options['score_func'],
                    options['weighting'], options['algorithm'], options['merging_bootstraps'], options['pruning_bootstraps'],
                    options['delta'], options['precis_threshold'], i)
                ))

            # block and collect the pool
            pool.close()
            pool.join()

            # get the async results and sort to ensure original batch index order and remove batch index
            CHIRPS_exps = [async_out[j].get() for j in range(len(async_out))]
            CHIRPS_exps.sort()
            for i in range(n_instances):
                CHIRPS_explainers[i] = CHIRPS_exps[i][1]  # return in list

        else:
            for i in range(n_instances):
                if i % 5 == 0: print('Working on CHIRPS for instance ' + str(i) + ' of ' + str(n_instances))
                # get a CHIRPS_runner per instance
                # filtering by the chosen set of trees - default: majority voting
                # use deepcopy to ensure by_value, not by_reference instantiation
                c_runner = self.bp_container.get_CHIRPS_runner(i, deepcopy(self.meta_data), which_trees=options['which_trees'], random_state=random_state)
                # run the chirps process on each instance paths
                _, CHIRPS_exp = \
                    as_CHIRPS(c_runner, target_classes[i],
                    self.sample_instances, self.sample_labels,
                    self.forest, self.fwmet,
                    options['paths_lengths_threshold'], options['support_paths'], options['alpha_paths'],
                    options['disc_path_bins'], options['disc_path_eqcounts'], options['score_func'],
                    options['weighting'], options['algorithm'], options['merging_bootstraps'], options['pruning_bootstraps'],
                    options['delta'], options['precis_threshold'], i)

                # add the finished rule accumulator to the results
                CHIRPS_explainers[i] = CHIRPS_exp

        self.CHIRPS_explainers = CHIRPS_explainers
