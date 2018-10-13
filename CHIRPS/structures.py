import sys
import math
import multiprocessing as mp
import numpy as np
from CHIRPS import p_count, p_count_corrected, if_nexists_make_dir
from pandas import DataFrame, Series
from pyfpgrowth import find_frequent_patterns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy import sparse
from scipy.stats import sem
from operator import itemgetter
from itertools import chain
from scipy.stats import chi2_contingency
from CHIRPS import config as cfg
from CHIRPS.async_structures import *

class default_encoder:

    def transform(x):
        return(sparse.csr_matrix(x))
    def fit(x):
        return(x)

# this is inherited by CHIRPS_runner and data_container
class non_deterministic:

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
class data_split_container:

    def __init__(self, X_train, X_train_enc,
                X_test, X_test_enc,
                y_train, y_test,
                train_priors, test_priors,
                train_index=None, test_index=None):
        self.X_train = X_train
        self.X_train_enc = X_train_enc
        self.X_test_enc = X_test_enc
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_priors = train_priors
        self.test_priors = test_priors

        self.X_train_matrix = np.matrix(X_train)
        self.X_test_matrix = np.matrix(X_test)

        to_mat = lambda x : x.todense() if isinstance(x, sparse.csr.csr_matrix) else x
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

    def get_next(self, batch_size = 1, encoded=False, which_split='train'):
        if which_split == 'train':
            instances = self.X_train[self.current_row_train:self.current_row_train + batch_size]
            instances_enc = self.X_train_enc[self.current_row_train:self.current_row_train + batch_size]
            instances_enc_matrix = self.X_train_enc_matrix[self.current_row_train:self.current_row_train + batch_size]
            labels = self.y_train[self.current_row_train:self.current_row_train + batch_size]
            self.current_row_train += batch_size
        else:
            instances = self.X_test[self.current_row_test:self.current_row_test + batch_size]
            instances_enc = self.X_test_enc[self.current_row_test:self.current_row_test + batch_size]
            instances_enc_matrix = self.X_test_enc_matrix[self.current_row_test:self.current_row_test + batch_size]
            labels = self.y_test[self.current_row_test:self.current_row_test + batch_size]
            self.current_row_test += batch_size
        return(instances, instances_enc, instances_enc_matrix, labels)

    # leave one out by instance_id and encode the rest
    def get_loo_instances(self, instance_id, encoded=False, which_split='test'):
        if which_split == 'train':
            instances = self.X_train.drop(instance_id)
            loc_index = self.y_train.index == instance_id
            instances_enc = self.X_train_enc[~loc_index,:]
            labels = self.y_train.drop(instance_id)
        else:
            instances = self.X_test.drop(instance_id)
            loc_index = self.y_test.index == instance_id
            instances_enc = self.X_test_enc[~loc_index,:]
            labels = self.y_test.drop(instance_id)
        return(instances, instances_enc, labels)

    def to_dict(self):
        return({'X_train': self.X_train,
            'X_train_enc' : self.X_train_enc,
            'X_train_enc_matrix' : self.X_train_enc_matrix,
            'X_test' : self.X_test,
            'X_test_enc' : self.X_test_enc,
            'X_test_enc_matrix' : self.X_test_enc_matrix,
            'y_train' : self.y_train,
            'y_test' : self.y_test,
            'train_priors' : self.train_priors,
            'test_priors' : self.test_priors,
            'train_index' : self.train_index,
            'test_index' : self.test_index})

    def indexes_to_csv(self, save_path = ''):
        if_nexists_make_dir(save_path)
        Series(self.train_index).to_csv(path = save_path + 'train_index.csv', index=False)
        Series(self.test_index).to_csv(path = save_path + 'test_index.csv', index=False)

    def data_to_csv(self, save_path = '', encoded_features = None):
        if_nexists_make_dir(save_path)
        self.X_train.to_csv(path_or_buf = save_path + 'X_train.csv')
        if encoded_features:
            DataFrame(self.X_train_enc.todense(),
                    columns=encoded_features).to_csv(
                                                    path_or_buf = save_path + 'X_train_enc.csv')
            DataFrame(self.X_test_enc.todense(),
                    columns=encoded_features).to_csv(
                                                    path_or_buf = save_path + 'X_test_enc.csv')
        self.y_train.to_csv(path = save_path + 'y_train.csv')
        self.X_test.to_csv(path_or_buf = save_path + 'X_test.csv')
        self.y_test.to_csv(path = save_path + 'y_test.csv')

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
                                , 'onehot_labels' : [v + '_' + str(n) for n in names] if t == 'nominal' else None
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
                self.features_enc.append(self.var_dict[f]['onehot_labels'])

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

    # helper function for pickling files
    def make_save_path(self, filename = ''):
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
            return self.le_dict[col].inverse_transform([label])[0]
        else:
            return(label)

    # generate indexes for manual tt_split
    def get_tt_split_idx(self, test_size=0.3, random_state=None, shuffle=True):
        # common default setting: see class non_deterministic
        random_state = self.default_if_none_random_state(random_state)
        nrows = self.data.shape[0]
        np.random.seed(random_state)
        test_idx = np.random.choice(nrows - 1, # zero base
                                    size = round(test_size * nrows),
                                    replace=False)
        # this method avoids scanning the array for each test_idx to find all the others
        train_pos = Series([True] * nrows)
        train_pos.loc[test_idx] = False
        train_idx = np.array(train_pos.index[train_pos], dtype=np.int32)
        # train are currently in given order, the test are not
        if shuffle:
            np.random.seed(random_state)
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

        # determine the priors for each split
        train_priors = y_train.value_counts().sort_index()/len(y_train)
        test_priors = y_test.value_counts().sort_index()/len(y_test)

        # create an encoded copy
        X_train_enc = self.encoder.transform(X_train)
        X_test_enc = self.encoder.transform(X_test)

        tt = data_split_container(X_train = X_train,
        X_train_enc = X_train_enc,
        X_test = X_test,
        X_test_enc = X_test_enc,
        y_train = y_train,
        y_test = y_test,
        train_priors = train_priors,
        test_priors = test_priors)

        return(tt)

    def get_meta(self):
        return({'class_col' : self.class_col,
                'class_names' : self.class_names,
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
                'random_state' : self.random_state
                })

# classes and functions for the parallelisable tree_walk

class instance_paths_container:
    def __init__(self
    , paths
    , tree_preds
    , patterns=None):
        self.paths = paths
        self.tree_preds = tree_preds
        self.patterns = patterns

    def discretize_paths(self, var_dict, bins=4, equal_counts=False):
        # check if bins is not numeric or can't be cast, then force equal width (equal_counts = False)
        if equal_counts:
            def hist_func(x, bins):
                npt = len(x)
                return np.interp(np.linspace(0, npt, bins + 1),
                                 np.arange(npt),
                                 np.sort(x))
        else:
            def hist_func(x, bins):
                return(np.histogram(x, bins))

        cont_vars = [vn for vn in var_dict if var_dict[vn]['data_type'] == 'continuous' and var_dict[vn]['class_col'] == False]
        for feature in cont_vars:
        # nan warnings OK, it just means the less than or greater than test was never used
            # lower bound, greater than
            lowers = [item[2] for nodes in self.paths for item in nodes if item[0] == feature and item[1] == False]

            # upper bound, less than
            uppers = [item[2] for nodes in self.paths for item in nodes if item[0] == feature and item[1] == True]

            upper_bins = np.histogram(uppers, bins=bins)[1]
            lower_bins = np.histogram(lowers, bins=bins)[1]

            # upper_bin_midpoints = pd.Series(upper_bins).rolling(window=2, center=False).mean().values[1:]
            upper_bin_means = (np.histogram(uppers, upper_bins, weights=uppers)[0] /
                                np.histogram(uppers, upper_bins)[0]).round(5)

            # lower_bin_midpoints = pd.Series(lower_bins).rolling(window=2, center=False).mean().values[1:]
            lower_bin_means = (np.histogram(lowers, lower_bins, weights=lowers)[0] /
                                np.histogram(lowers, lower_bins)[0]).round(5)

            # discretize functions from histogram means
            upper_discretize = lambda x: upper_bin_means[np.max([np.min([np.digitize(x, upper_bins), len(upper_bin_means)]), 1]) - 1]
            lower_discretize = lambda x: lower_bin_means[np.max([np.min([np.digitize(x, lower_bins, right= True), len(upper_bin_means)]), 1]) - 1]

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

    def mine_patterns(self, support=0.1):
        # convert to an absolute number of instances rather than a fraction
        if support < 1:
            support = round(support * len(self.paths))
        self.patterns = find_frequent_patterns(self.paths, support)

    def sort_patterns(self, alpha=0.0, weights=None):
        alpha = float(alpha)
        if weights is None:
            weights = [1] * len(self.patterns)
        fp_scope = self.patterns.copy()
        # to shrink the support of shorter freq_patterns
        # formula is sqrt(weight) * log(sup * (len - alpha) / len)
        score_function = lambda x, w: (x[0], x[1], math.sqrt(w) * math.log(x[1]) * (len(x[0]) - alpha) / len(x[0]))
        fp_scope = [fp for fp in map(score_function, fp_scope.items(), weights)]
        # score is now at position 2 of tuple
        self.patterns = sorted(fp_scope, key=itemgetter(2), reverse = True)

class batch_paths_container:
    def __init__(self
    , path_detail
    , by_tree):
        self.by_tree = by_tree # given as initial state, not used to reshape on construction. use flip() if necessary
        self.path_detail = path_detail

    def flip(self):
        n = len(self.path_detail[0])
        flipped_paths = [[]] * n
        for i in range(n):
            flipped_paths[i] =  [p[i] for p in self.path_detail]
        self.path_detail = flipped_paths
        self.by_tree = not self.by_tree

    def major_class_from_paths(self, batch_idx, return_counts=True):
        if self.by_tree:
            pred_classes = [self.path_detail[p][batch_idx]['pred_class'] for p in range(len(self.path_detail))]
        else:
            pred_classes = [self.path_detail[batch_idx][p]['pred_class'] for p in range(len(self.path_detail[batch_idx]))]

        unique, counts = np.unique(pred_classes, return_counts=True)

        if return_counts:
            return(unique[np.argmax(counts)], dict(zip(unique, counts)))
        else: return(unique[np.argmax(counts)])

    def get_instance_paths(self, batch_idx, which_trees='all', feature_values=True):
        # It is a sequence number in the batch of instance_paths that have been walked
        # Each path is the route of one instance down one tree

        batch_idx = math.floor(batch_idx) # make sure it's an integer
        true_to_lt = lambda x: '<' if x == True else '>'

        # extract the paths we want by filtering on tree performance
        if self.by_tree:
            n_paths = len(self.path_detail)
            if which_trees == 'correct':
                paths_info = [self.path_detail[pd][batch_idx]['path'] for pd in range(n_paths) if self.path_detail[pd][batch_idx]['tree_agree_maj_vote']]
            elif which_trees == 'majority':
                major_class = self.major_class_from_paths(batch_idx, return_counts=False)
                paths_info = [self.path_detail[pd][batch_idx]['path'] for pd in range(n_paths) if self.path_detail[pd][batch_idx]['pred_class'] == major_class]
            elif which_trees == 'minority':
                major_class = self.major_class_from_paths(batch_idx, return_counts=False)
                paths_info = [self.path_detail[pd][batch_idx]['path'] for pd in range(n_paths) if self.path_detail[pd][batch_idx]['pred_class'] != major_class]
            else:
                paths_info = [self.path_detail[pd][batch_idx]['path'] for pd in range(n_paths)]
        else:
            n_paths = len(self.path_detail[batch_idx])
            if which_trees == 'correct':
                paths_info = [self.path_detail[batch_idx][pd]['path'] for pd in range(n_paths) if self.path_detail[batch_idx][pd]['tree_agree_maj_vote']]
            elif which_trees == 'majority':
                major_class = self.major_class_from_paths(batch_idx, return_counts=False)
                paths_info = [self.path_detail[batch_idx][pd]['path'] for pd in range(n_paths) if self.path_detail[batch_idx][pd]['pred_class'] == major_class]
            elif which_trees == 'minority':
                major_class = self.major_class_from_paths(batch_idx, return_counts=False)
                paths_info = [self.path_detail[batch_idx][pd]['path'] for pd in range(n_paths) if self.path_detail[batch_idx][pd]['pred_class'] != major_class]
            else:
                paths_info = [self.path_detail[batch_idx][pd]['path'] for pd in range(n_paths)]

        # path formatting - should it be on values level or features level
        if feature_values:
            paths = [[]] * len(paths_info)
            for i, p in enumerate(paths_info):
                paths[i] = [(f, leq, t) for f, leq, t in zip(p['feature_name'], p['leq_threshold'], p['threshold'])]
        else:
            paths = [p['feature_name'] for p in paths_info]

        # tree performance stats
        if self.by_tree:
            tree_preds = [self.path_detail[t][batch_idx]['pred_class_label'] for t in range(n_paths)]
        else:
            tree_preds = [self.path_detail[batch_idx][t]['pred_class_label'] for t in range(n_paths)]

        # return an object for requested instance
        instance_paths = instance_paths_container(paths, tree_preds)
        return(instance_paths)

class forest_walker:

    def __init__(self
    , forest
    , meta_data):
        self.forest = forest
        self.features = meta_data['features_enc']
        self.n_features = len(self.features)
        self.class_col = meta_data['class_col']

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
        self.structure = {'root_features' : self.root_features
                         , 'child_features' : self.child_features
                         , 'lower_features' : self.lower_features}

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
        self.n_classes = len(np.unique(labels))

        if labels is not None:
            if len(labels) != self.n_instances:
                raise ValueError("labels and instances must be same length")

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

        return(tree_pred, tree_pred_labels, tree_pred_proba, tree_agree_maj_vote, feature, threshold, path)

    def forest_walk(self, instances, labels = None, forest_walk_async=False):

        features = self.features
        n_instances = instances.shape[0]

        if forest_walk_async:
            async_out = []
            n_cores = mp.cpu_count()-1
            pool = mp.Pool(processes=n_cores)

            for i, t in enumerate(self.forest.estimators_):

                # process the tree
                tree_pred, tree_pred_labels, \
                tree_pred_proba, tree_agree_maj_vote, \
                feature, threshold, path = self.tree_structures(t, instances, labels, n_instances)
                # walk the tree
                async_out.append(pool.apply_async(as_tree_walk,
                                                (i, instances, labels, n_instances,
                                                tree_pred, tree_pred_labels,
                                                tree_pred_proba, tree_agree_maj_vote,
                                                feature, threshold, path, features)
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
            for i, t in enumerate(self.forest.estimators_):

                # process the tree
                tree_pred, tree_pred_labels, \
                tree_pred_proba, tree_agree_maj_vote, \
                feature, threshold, path = self.tree_structures(t, instances, labels, n_instances)
                # walk the tree
                _, tree_paths[i] = as_tree_walk(i, instances, labels, n_instances,
                                                tree_pred, tree_pred_labels,
                                                tree_pred_proba, tree_agree_maj_vote,
                                                feature, threshold, path, features)

        return(batch_paths_container(tree_paths, by_tree=True))

# classes and functions for the parallelisable CHIRPS algorithm

# this is inherited by CHIRPS_explainer and CHIRPS_runner
# the main point is to have the rule_evaluator function inherited from one place
class rule_evaluator:

    def init_values(self, rule=None, features=None, class_names=None,
                    instances=None, labels=None):

        # sub-classes must have these three properties
        if rule is None:
                rule = self.rule
        if features is None:
            features = self.features_enc # default
        if class_names is None:
            class_names = self.class_names

        # check presence of optional sample datasets:
        # train (or other) for optimisation of rule merge
        # test (or other) for evaluation of rule
        if instances is None:
            if self.sample_instances is None:
                print('Sample intances (e.g. X_train_enc) are required for rule evaluation')
                return()
            else:
                instances = self.sample_instances
        if labels is None:
            if self.sample_labels is None:
                print('Sample labels (e.g. y_train) are required for rule evaluation')
                return()
            else:
                labels = self.sample_labels

        return(rule, features, class_names, instances, labels)

    # apply a rule on an instance space, returns covered instance idx
    def apply_rule(self, rule=None, instances=None, features=None):

        lt_gt = lambda x, y, z : x <= y if z else x > y # if z is True, x <= y else x > y
        idx = np.full(instances.shape[0], 1, dtype='bool')
        for r in rule:
            idx = np.logical_and(idx, lt_gt(instances.getcol(features.index(r[0])).toarray().flatten(), r[2], r[1]))
        return(idx)

    # score a rule on an instance space
    def evaluate_rule(self, rule=None, features=None, class_names=None,
                        instances=None, labels=None):


        rule, features, class_names, instances, labels = \
                self.init_values(rule=rule, features=features, class_names=class_names,
                                instances=instances, labels=labels)

        idx = self.apply_rule(rule, instances, features)
        coverage = idx.sum()/len(idx) # tp + fp / tp + fp + tn + fn

        priors = p_count_corrected(labels, [i for i in range(len(class_names))])

        p_counts = p_count_corrected(labels.iloc[idx], [i for i in range(len(class_names))]) # true positives
        post = p_counts['p_counts']
        p_corrected = np.array([p if p > 0.0 else 1.0 for p in post]) # to avoid div by zeros

        counts = p_counts['counts']
        labels = p_counts['labels']

        observed = np.array((counts, priors['counts']))
        if counts.sum() > 0: # previous_counts.sum() == 0 is impossible
            chisq = chi2_contingency(observed=observed[:, np.where(observed.sum(axis=0) != 0)], correction=True)
        else:
            chisq = None

        # class coverage, TPR (recall) TP / (TP + FN)
        recall = counts / priors['counts']
        r_corrected = np.array([r if r > 0.0 else 1.0 for r in recall]) # to avoid div by zeros
        f1 = [2] * ((post * recall) / (p_corrected + r_corrected))

        not_covered_counts = counts + (np.sum(priors['counts']) - priors['counts']) - (np.sum(counts) - counts)
        # accuracy = (TP + TN) / num_instances formula: https://books.google.co.uk/books?id=ubzZDQAAQBAJ&pg=PR75&lpg=PR75&dq=rule+precision+and+coverage&source=bl&ots=Aa4Gj7fh5g&sig=6OsF3y4Kyk9KlN08OPQfkZCuZOc&hl=en&sa=X&ved=0ahUKEwjM06aW2brZAhWCIsAKHY5sA4kQ6AEIUjAE#v=onepage&q=rule%20precision%20and%20coverage&f=false
        accu = not_covered_counts/priors['counts'].sum()

        # to avoid div by zeros
        pri_corrected = np.array([pri if pri > 0.0 else 1.0 for pri in priors['p_counts']])
        pos_corrected = np.array([pos if pri > 0.0 else 0.0 for pri, pos in zip(priors['p_counts'], post)])
        if counts.sum() == 0:
            rec_corrected = np.array([0.0] * len(pos_corrected))
            cov_corrected = np.array([1.0] * len(pos_corrected))
        else:
            rec_corrected = counts / counts.sum()
            cov_corrected = np.array([counts.sum() / priors['counts'].sum()])

        # lift = precis / (total_cover * prior)
        lift = pos_corrected / ( ( cov_corrected ) * pri_corrected )

        return({'coverage' : coverage,
                'priors' : priors,
                'post' : post,
                'counts' : counts,
                'labels' : labels,
                'recall' : recall,
                'f1' : f1,
                'accuracy' : accu,
                'lift' : lift,
                'chisq' : chisq})

class CHIRPS_explainer(rule_evaluator):

    def __init__(self, features, features_enc, class_names,
                var_dict, var_dict_enc,
                paths, patterns,
                rule, pruned_rule, conjunction_rule,
                target_class, target_class_label,
                major_class, major_class_label,
                model_votes, model_posterior,
                coverage, precision, posterior,
                accuracy,
                counts,
                recall,
                f1,
                lift,
                algorithm):
        self.features = features
        self.features_enc = features_enc
        self.class_names = class_names
        self.var_dict = var_dict
        self.var_dict_enc = var_dict_enc
        self.paths = paths
        self.patterns = patterns
        self.rule = rule
        self.pruned_rule = pruned_rule
        self.conjunction_rule = conjunction_rule
        self.target_class = target_class
        self.target_class_label = target_class_label
        self.major_class = major_class
        self.major_class_label = major_class_label
        self.model_votes = model_votes
        self.model_posterior = model_posterior
        self.coverage = coverage
        self.precision = precision
        self.posterior = posterior
        self.accuracy = accuracy
        self.counts = counts
        self.recall = recall
        self.f1 = f1
        self.lift = lift
        self.algorithm = algorithm

    def prettify_rule(self, rule=None, var_dict=None):

        if rule is None: # default
            rule = self.pruned_rule

        if var_dict is None: # default - match prediction model
            var_dict = self.var_dict_enc

        Tr_Fa = lambda x, y, z : x + ' True' if ~y else x + ' False'
        lt_gt = lambda x, y, z : x + ' <= ' + str(z) if y else x + ' > ' + str(z)
        def bin_or_cont(x, y, z):
            if x in var_dict:
                return(Tr_Fa(x,y,z))
            else:
                return(lt_gt(x,y,z))
        return(' AND '.join([bin_or_cont(f, t, v) for f, t, v in rule]))

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
        'conjunction_rule' : self.conjunction_rule,
        'target_class' :self.target_class,
        'target_class_label' :self.target_class_label,
        'major_class' : self.major_class,
        'major_class_label' :self.major_class_label,
        'model_votes' : self.model_votes,
        'model_posterior' : self.model_posterior,
        'coverage' : self.coverage,
        'precision' : self.precision,
        'posterior' : self.posterior,
        'accuracy' : self.accuracy,
        'counts' : self.counts,
        'recall' : self.recall,
        'f1' : self.f1,
        'lift' : self.lift,
        'algorithm' : algorithm})

# this class runs all steps of the CHIRPS algorithm
class CHIRPS_runner(non_deterministic, rule_evaluator):

    def __init__(self, instance_paths_container, meta_data):

        meta_random_state = meta_data.get('random_state')
        if meta_random_state is not None:
            self.random_state = meta_random_state # otherwise there is a default

        self.ip_container = instance_paths_container

        self.features = meta_data['features']
        self.features_enc = meta_data['features_enc']
        self.var_dict = meta_data['var_dict']
        self.var_dict_enc = meta_data['var_dict_enc']
        self.class_col = meta_data['class_col']

        meta_le_dict = meta_data['le_dict']
        meta_get_label = meta_data['get_label']
        meta_class_names = meta_data['class_names']
        if self.class_col in meta_le_dict.keys():
            self.class_names = meta_get_label(self.class_col, [i for i in range(len(meta_class_names))])
            self.get_label = meta_get_label
        else:
            self.class_names = meta_class_names
            self.get_label = None

        self.model_votes = p_count_corrected(self.ip_container.tree_preds, self.class_names)

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
        self.conjunction_rule = []
        self.__previous_rule = []
        self.__reverted = []
        self.total_points = None
        self.accumulated_points = 0
        self.sample_instances = None
        self.sample_labels = None
        self.n_instances = None
        self.n_classes = None
        self.target_class = None
        self.target_class_label = None
        self.major_class = None
        self.model_entropy = None
        self.model_info_gain = None
        self.model_posterior = None
        self.max_ent = None
        self.coverage = None
        self.precision = None
        self.cum_info_gain = None
        self.information_gain = None
        self.prior_entropy = None
        self.prior_info = None
        self.posterior = None
        self.accuracy = None
        self.counts = None
        self.recall = None
        self.f1 = None
        self.lift = None
        self.isolation_pos = None
        self.stopping_param = None
        self.merge_rule_iter = None
        self.algorithm = None

    def add_rule_term(self, p_total = 0.1):
        self.__previous_rule = deepcopy(self.rule)
        next_rule = self.patterns[self.unapplied_rules[0]]
        for item in next_rule[0]:
            if item in self.rule:
                continue # skip duplicates (essential for pruning reasons)
            if item[0] in self.var_dict_enc: # binary feature
                # update the master list
                position = self.var_dict[self.var_dict_enc[item[0]]]['onehot_labels'].index(item[0])
                if item[1]: # leq_threshold True
                    self.var_dict[self.var_dict_enc[item[0]]]['upper_bound'][position] = item[2]
                else:
                    self.var_dict[self.var_dict_enc[item[0]]]['lower_bound'][position] = item[2]
                # append or update
                self.rule.append(item)

            else: # continuous feature
                append_or_update = False
                if item[1]: # leq_threshold True
                    if item[2] <= self.var_dict[item[0]]['upper_bound'][0]:
                        self.var_dict[item[0]]['upper_bound'][0] = item[2]
                        append_or_update = True

                else:
                    if item[2] > self.var_dict[item[0]]['lower_bound'][0]:
                        self.var_dict[item[0]]['lower_bound'][0] = item[2]
                        append_or_update = True

                if append_or_update:
                    feature_appears = [(f, ) for (f, t, _) in self.rule]
                    if (item[0],) in feature_appears:
                        # print(item, 'feature appears already')
                        valueless_rule = [(f, t) for (f, t, _) in self.rule]
                        if (item[0], item[1]) in valueless_rule: # it's already there and needs updating
                            # print(item, 'feature values appears already')
                            self.rule[valueless_rule.index((item[0], item[1]))] = item
                        else: # feature has been used at the opposite end (either lower or upper bound) and needs inserting
                            # print(item, 'feature values with new discontinuity')
                            self.rule.insert(feature_appears.index((item[0],)) + 1, item)
                    else:
                        # print(item, 'feature first added')
                        self.rule.append(item)

            # accumlate points from rule and tidy up
            # remove the first item from unapplied_rules as it's just been applied or ignored for being out of range
            self.accumulated_points += self.patterns[0][2]
            del self.unapplied_rules[0]
            # accumlate all the freq patts that are subsets of the current rules
            # remove the index from the unapplied rules list (including the current rule just added)
            to_remove = []
            for ur in self.unapplied_rules:
                # check if all items are already part of the rule (i.e. it's a subset)
                if all([item in self.rule for item in self.patterns[ur][0]]):
                    self.accumulated_points += self.patterns[ur][2]
                    # collect up the values to remove. don't want to edit the iterator in progress
                    to_remove.append(ur)
            for rmv in reversed(to_remove):
                self.unapplied_rules.remove(rmv)

    def prune_rule(self):
        # remove all other binary items if one Greater than is found.
        gt_items = {} # find all the items with the leq_threshold False
        for item in self.rule:
            if ~item[1] and item[0] in self.var_dict_enc: # item is greater than thresh and a nominal type
                gt_items[self.var_dict_enc[item[0]]] = item[0] # capture the parent feature and the feature value

        gt_pruned_rule = []
        for item in self.rule:
            if item[0] in self.var_dict_enc: # binary variable
                if self.var_dict_enc[item[0]] not in gt_items.keys():
                    gt_pruned_rule.append(item)
                elif ~item[1]:
                    gt_pruned_rule.append(item)
            else: # leave continuous as is
                gt_pruned_rule.append(item)

        # if all but one of a feature set is False, swap them out for the remaining value
        # start by counting all the lt thresholds in each parent feature
        lt_items = defaultdict(lambda: 0)
        for item in gt_pruned_rule: # find all the items with the leq_threshold True
            if item[1] and item[0] in self.var_dict_enc: # item is less than thresh and a nominal type
                lt_items[self.var_dict_enc[item[0]]] += 1 # capture the parent feature and count each

        # checking if just one other feature value remains unused
        pruned_items = [item[0] for item in gt_pruned_rule]
        for lt in dict(lt_items).keys():
            n_categories = len([i for i in self.var_dict_enc.values() if i == lt])
            if n_categories - dict(lt_items)[lt] == 1:
                # get the remaining value for this feature
                lt_labels = self.var_dict[lt]['onehot_labels']
                to_remove = [label for label in lt_labels if label in pruned_items]
                remaining_value = [label for label in lt_labels if label not in pruned_items]

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

        # find a rule with only nominal features are only binary True. Always include numeric
        self.conjunction_rule = [r for r in self.pruned_rule if not r[1] or self.var_dict[r[0]]['data_type'] == 'continuous']

    def __greedy_commit__(self, current, previous):
        if current <= previous:
            self.rule = deepcopy(self.__previous_rule)
            self.__reverted.append(True)
            return(True)
        else:
            self.__reverted.append(False)
            return(False)

    def merge_rule(self, sample_instances, sample_labels, forest
                        , stopping_param = 1
                        , precis_threshold = 1.0
                        , fixed_length = None
                        , target_class = None
                        , algorithm='greedy_prec'
                        , random_state=None):

        self.paths = self.ip_container.paths
        self.patterns = self.ip_container.patterns
        self.unapplied_rules = [i for i in range(len(self.patterns))]
        self.total_points = sum([scrs[2] for scrs in self.patterns])

        # basic setup
        # pointless to receive a None for algorithm
        if algorithm is None:
            self.algorithm = 'greedy_prec'
        else:
            self.algorithm = algorithm
        # common default setting: see class non_deterministic
        random_state = self.default_if_none_random_state(random_state)
        if stopping_param > 1 or stopping_param < 0:
            stopping_param = 1
            print('warning: stopping_param should be 0 <= p <= 1. Value was reset to 1')
        self.stopping_param = stopping_param
        self.sample_instances = sample_instances
        self.sample_labels = sample_labels
        self.n_classes = len(np.unique(self.sample_labels))
        self.n_instances = len(self.sample_labels)

        # model posterior
        # model votes collected in constructor
        self.model_posterior = self.model_votes['p_counts']

        # model final entropy
        self.model_entropy = entropy(self.model_posterior)

        # model predicted class
        self.major_class = np.argmax(self.model_posterior)
        if self.get_label is None:
            self.major_class_label = self.major_class
        else:
            self.major_class_label = self.get_label(self.class_col, self.major_class)

        # this analysis
        # target class
        if target_class is None:
            self.target_class = self.major_class
            self.target_class_label = self.major_class_label
        else:
            self.target_class = target_class
            if self.get_label is None:
                self.target_class_label = self.target_class
            else:
                self.target_class_label = self.get_label(self.class_col, self.target_class)

        # prior - empty rule
        p_counts = p_count_corrected(sample_labels.values, [i for i in range(len(self.class_names))])
        self.posterior = np.array([p_counts['p_counts'].tolist()])
        self.counts = np.array([p_counts['counts'].tolist()])
        self.recall = [np.full(self.n_classes, 1.0)] # counts / prior counts
        self.f1 =  [2] * ( ( self.posterior * self.recall ) / ( self.posterior + self.recall ) ) # 2 * (precis * recall/(precis + recall) )
        self.accuracy = np.array([p_counts['p_counts'].tolist()])
        self.lift = [np.full(self.n_classes, 1.0)] # precis / (total_cover * prior)
        self.prior_entropy = entropy(self.counts[0])

        # info gain
        self.max_ent = entropy([1 / self.n_classes] * self.n_classes)
        self.model_info_gain = self.max_ent - self.model_entropy
        self.prior_info = self.max_ent - self.prior_entropy

        # pre-loop set up
        # rule based measures - prior/empty rule
        current_precision = p_counts['p_counts'][np.where(p_counts['labels'] == self.target_class)][0] # based on priors

        self.coverage = [1.0]
        self.precision = [current_precision]

        # rule posteriors
        previous_entropy = self.max_ent # start at max possible
        current_entropy = self.prior_entropy # entropy of prior distribution
        self.information_gain = [previous_entropy - current_entropy] # information baseline (gain of priors over maximum)
        self.cum_info_gain = self.information_gain.copy()

        # accumulate rule terms
        cum_points = 0
        self.merge_rule_iter = 0

        while current_precision != 1.0 and current_precision != 0.0 and current_precision < precis_threshold and self.accumulated_points <= self.total_points * self.stopping_param and (fixed_length is None or len(self.cum_info_gain) < max(1, fixed_length) + 1):
            self.merge_rule_iter += 1
            self.add_rule_term(p_total = self.stopping_param)
            # you could add a round of bootstrapping here, but what does that do to performance
            eval_rule = self.evaluate_rule(instances=sample_instances,
                                            labels=sample_labels)

            # entropy / information
            previous_entropy = current_entropy
            current_entropy = entropy(eval_rule['post'])

            # code to confirm rule, or revert to previous
            # choosing from a range of possible metrics and learning improvement
            # possible to introduce annealing?
            # e.g if there was no change, or an decrease in precis
            if self.algorithm == 'greedy_prec':
                current = eval_rule['post'][np.where(eval_rule['labels'] == self.target_class)]
                previous = list(reversed(self.posterior))[0][np.where(eval_rule['labels'] == self.target_class)]
                should_continue = self.__greedy_commit__(current, previous)
            elif self.algorithm == 'greedy_f1':
                current = eval_rule['f1'][np.where(eval_rule['labels'] == self.target_class)]
                previous = list(reversed(self.f1))[0][np.where(eval_rule['labels'] == self.target_class)]
                should_continue = self.__greedy_commit__(current, previous)
            elif self.algorithm == 'greedy_acc':
                current = eval_rule['accuracy'][np.where(eval_rule['labels'] == self.target_class)]
                previous = list(reversed(self.accuracy))[0][np.where(eval_rule['labels'] == self.target_class)]
                should_continue = self.__greedy_commit__(current, previous)
            elif self.algorithm == 'chi2':
                previous_counts = list(reversed(self.counts))[0]
                observed = np.array((eval_rule['counts'], previous_counts))
                if eval_rule['counts'].sum() == 0: # previous_counts.sum() == 0 is impossible
                    should_continue = self.__greedy_commit__(1, 0) # go ahead with rule as the algorithm will finish here
                else: # do the chi square test but mask any classes where prev and current are zero
                    should_continue = self.__greedy_commit__(0.05, chi2_contingency(observed=observed[:, np.where(observed.sum(axis=0) != 0)], correction=True)[1])
            # add more options here
            else: should_continue = False
            if should_continue:
                continue # don't update all the metrics, just go to the next round

            # check for end conditions; no target class coverage
            if eval_rule['post'][np.where(eval_rule['labels'] == self.target_class)] == 0.0:
                current_precision = 0.0
            else:
                current_precision = eval_rule['post'][np.where(eval_rule['labels'] == self.target_class)][0]

            # if we keep the new rule, append the results to the persisted arrays
            # general coverage and precision
            self.precision.append(current_precision)
            self.coverage.append(eval_rule['coverage'])

            # per class measures
            self.posterior = np.append(self.posterior, [eval_rule['post']], axis=0)
            self.counts = np.append(self.counts, [eval_rule['counts']], axis=0)
            self.accuracy = np.append(self.accuracy, [eval_rule['accuracy']], axis=0)
            self.recall = np.append(self.recall, [eval_rule['recall']], axis=0 )
            self.f1 = np.append(self.f1, [eval_rule['f1']], axis=0 )
            self.lift = np.append(self.lift, [eval_rule['lift']], axis=0 )

            # entropy and info gain
            self.information_gain.append(previous_entropy - current_entropy)
            self.cum_info_gain.append(sum(self.information_gain))

        # first time major_class is isolated
        if any(np.argmax(self.posterior, axis=1) == self.target_class):
            self.isolation_pos = np.min(np.where(np.argmax(self.posterior, axis=1) == self.target_class))
        else: self.isolation_pos = None

    def score_rule(self, alpha=0.5):
        target_precision = [p[self.target_class] for p in self.posterior]
        target_recall = [r[self.target_class] for r in self.recall]
        target_f1 = [f[self.target_class] for f in self.f1]
        target_accuracy = [a[self.target_class] for a in self.accuracy]
        target_prf = [[p, r, f, a] for p, r, f, a in zip(target_precision, target_recall, target_f1, target_accuracy)]

        target_cardinality = [i for i in range(len(target_precision))]

        lf = lambda x: math.log2(x + 1)
        score_fun1 = lambda f, crd, alp: lf(f * crd * alp / (1.0 + ((1 - alp) * crd**2)))
        score_fun2 = lambda a, crd, alp: lf(a * crd * alp / (1.0 + ((1 - alp) * crd**2)))

        score1 = [s for s in map(score_fun1, target_f1, target_cardinality, [alpha] * len(target_cardinality))]
        score2 = [s for s in map(score_fun2, target_accuracy, target_cardinality, [alpha] * len(target_cardinality))]

        return(target_prf, score1, score2)

    def mine_path_segments(self, support_paths=0.1, alpha_paths=0.5,
                            disc_path_bins=4, disc_path_eqcounts=False):

        # discretize any numeric features
        self.ip_container.discretize_paths(self.var_dict,
                                bins=disc_path_bins,
                                equal_counts=disc_path_eqcounts)
        # the patterns are found but not scored and sorted yet
        self.ip_container.mine_patterns(support=support_paths)

    def score_sort_path_segments(self, sample_instances, sample_labels,
                                    alpha_paths=0.5, weighting='chisq'):
        # best at -1 < alpha < 1
        # the patterns will be weighted by chi**2 for independence test, p-values
        if weighting == 'chisq':
            weights = [] * len(self.ip_container.patterns)
            for wp in self.ip_container.patterns:

                idx = self.apply_rule(rule=wp, instances=sample_instances, features=self.features_enc)
                covered = p_count_corrected(sample_labels[idx], [i for i in range(len(self.class_names))])['counts']
                not_covered = p_count_corrected(sample_labels[~idx], [i for i in range(len(self.class_names))])['counts']
                observed = np.array((covered, not_covered))

                # this is the chisq based weighting. can add other options
                if covered.sum() > 0 and not_covered.sum() > 0: # previous_counts.sum() == 0 is impossible
                    weights.append(math.sqrt(chi2_contingency(observed=observed[:, np.where(observed.sum(axis=0) != 0)], correction=True)[0]))
                else:
                    weights.append(max(weights))

            # now the patterns are scored and sorted. alpha > 0 favours longer patterns. 0 neutral. < 0 shorter.
            self.ip_container.sort_patterns(alpha=alpha_paths, weights=weights) # with chi2 and support sorting
        else:
            self.ip_container.sort_patterns(alpha=alpha_paths) # with only support/alpha sorting

    def get_CHIRPS_explainer(self):
        return(CHIRPS_explainer(self.features, self.features_enc, self.class_names,
        self.var_dict, self.var_dict_enc,
        self.paths, self.patterns,
        self.rule, self.pruned_rule, self.conjunction_rule,
        self.target_class, self.target_class_label,
        self.major_class, self.major_class_label,
        self.model_votes, self.model_posterior,
        self.coverage, self.precision, self.posterior,
        self.accuracy,
        self.counts,
        self.recall,
        self.f1,
        self.lift,
        self.algorithm))

class batch_CHIRPS_explainer:

    def __init__(self, bp_container, # batch_paths_container
                        forest, sample_instances, sample_labels, meta_data):
        self.bp_container = bp_container
        self.data_container = data_container
        self.forest = forest
        self.sample_instances = sample_instances
        self.sample_labels = sample_labels
        self.meta_data = meta_data

    def batch_run_CHIRPS(self, support_paths=0.1, alpha_paths=0.5,
                        disc_path_bins=4, disc_path_eqcounts=False,
                        alpha_scores=0.5, which_trees='majority',
                        precis_threshold=0.95, weighting='chisq',
                        algorithm='greedy_prec', chirps_explanation_async=False):

        # convenience function to orient the top level of bpc
        # a bit like reshaping an array
        # reason: rf paths quickly extracted per tree for all instances
        # so when constructed, this structure is oriented by tree
        # and we would like to easily iterate by instance
        if self.bp_container.by_tree:
            self.bp_container.flip()
        n_instances = len(self.bp_container.path_detail)

        # initialise a list for the results
        CHIRPS_explainers = [[]] * n_instances

        # generate the explanations
        if chirps_explanation_async:

            async_out = []
            n_cores = mp.cpu_count()-1
            pool = mp.Pool(processes=n_cores)

            # loop for each instance
            for i in range(n_instances):

                # extract the current instance paths container for freq patt mining
                # filtering by the chosen set of trees - default: majority voting
                ip_container = self.bp_container.get_instance_paths(i, which_trees=which_trees)
                c_runner = CHIRPS_runner(ip_container, self.meta_data)

                # run the chirps process on each instance paths
                async_out.append(pool.apply_async(as_CHIRPS,
                    (ip_container, c_runner,
                    self.sample_instances, self.sample_labels,
                    self.forest,
                    support_paths, alpha_paths,
                    disc_path_bins, disc_path_eqcounts,
                    weighting, algorithm, precis_threshold, i)
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
                # get the current individual_paths_container
                # filtering by the chosen set of trees - default: majority voting
                ip_container = self.bp_container.get_instance_paths(i, which_trees=which_trees)
                c_runner = CHIRPS_runner(ip_container, self.meta_data)

                # run the chirps process on each instance paths
                _, CHIRPS_exp = \
                    as_CHIRPS(ip_container, c_runner,
                    self.sample_instances, self.sample_labels,
                    self.forest,
                    support_paths, alpha_paths,
                    disc_path_bins, disc_path_eqcounts,
                    weighting, algorithm, precis_threshold, i)

                # add the finished rule accumulator to the results
                CHIRPS_explainers[i] = CHIRPS_exp

        self.CHIRPS_explainers = CHIRPS_explainers
