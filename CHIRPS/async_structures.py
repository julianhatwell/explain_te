# this module is required for parallel processing
# parallel requires functions/classes to be in __main__ or already referenced.
# as this code is quite complex, the latter is preferred
import math
import numpy as np
from scipy import sparse
from copy import deepcopy
from collections import deque, defaultdict
from scipy.stats import chi2_contingency, entropy
from CHIRPS import p_count, p_count_corrected

# parallelisable function for the forest_walker class
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

# classes and functions for the parallelisable CHIRPS algorithm

# this is inherited by rule_accumulator and data_container
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

# this is inherited by rule_accumulator, CHIRPS_container and CHIRPS_runner
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

class CHIRPS_container(rule_evaluator):

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
class rule_accumulator(non_deterministic, rule_evaluator):

    def __init__(self, data_container, instance_paths_container):

        self.random_state = data_container.random_state
        self.features_enc = data_container.features_enc
        self.var_dict_enc = data_container.var_dict_enc
        self.features = data_container.features
        self.var_dict = deepcopy(data_container.var_dict)
        self.paths = instance_paths_container.paths
        self.patterns = instance_paths_container.patterns
        self.unapplied_rules = [i for i in range(len(self.patterns))]

        self.class_col = data_container.class_col
        if data_container.class_col in data_container.le_dict.keys():
            self.class_names = data_container.get_label(data_container.class_col, [i for i in range(len(data_container.class_names))])
            self.get_label = data_container.get_label
        else:
            self.class_names = data_container.class_names
            self.get_label = None

        self.model_votes = p_count_corrected(instance_paths_container.tree_preds, self.class_names)

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
        self.total_points = sum([scrs[2] for scrs in self.patterns])
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

    def get_CHIRPS_container(self):
        return(CHIRPS_container(self.features, self.features_enc, self.class_names,
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

class CHIRPS_runner(rule_evaluator):

    def __init__(self, ip_container, data_container):
        self.ip_container = ip_container
        self.data_container = data_container

    def mine_path_segments(self, support_paths=0.1, alpha_paths=0.5,
                            disc_path_bins=4, disc_path_eqcounts=False):

        # discretize any numeric features
        self.ip_container.discretize_paths(self.data_container.var_dict,
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

                idx = self.apply_rule(rule=wp, instances=sample_instances, features=self.data_container.features_enc)
                covered = p_count_corrected(sample_labels[idx], [i for i in range(len(self.data_container.class_names))])['counts']
                not_covered = p_count_corrected(sample_labels[~idx], [i for i in range(len(self.data_container.class_names))])['counts']
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

    def get_rule(self, rule_acc, sample_instances, sample_labels, forest,
                            algorithm='greedy_prec', precis_threshold=0.95):

            # run the rule accumulator with greedy precis
            rule_acc.merge_rule(sample_instances=sample_instances,
                        sample_labels=sample_labels,
                        forest=forest,
                        algorithm=algorithm,
                        precis_threshold=precis_threshold)
            rule_acc.prune_rule()
            CHIRPS_cont = rule_acc.get_CHIRPS_container()

            # collect completed rule accumulator
            return(CHIRPS_cont)

def as_chirps(ip_container, data_container,
                        sample_instances, sample_labels, forest,
                        support_paths=0.1, alpha_paths=0.5,
                        disc_path_bins=4, disc_path_eqcounts=False,
                        weighting='chisq', algorithm='greedy_prec',
                        precis_threshold=0.95, batch_idx=None):
    # these steps make up the CHIRPS process:
    # mine paths for freq patts

    cr = CHIRPS_runner(ip_container, data_container)
    # fp growth mining
    cr.mine_path_segments(support_paths, alpha_paths,
                            disc_path_bins, disc_path_eqcounts)

    # score and sort
    cr.score_sort_path_segments(sample_instances, sample_labels,
                                    alpha_paths, weighting)

    ip_container = cr.ip_container
    # greedily add terms to create rule
    rule_acc = rule_accumulator(data_container=data_container, instance_paths_container=ip_container)
    CHIRPS_cont = cr.get_rule(rule_acc, sample_instances, sample_labels, forest,
    algorithm, precis_threshold)

    return(batch_idx, CHIRPS_cont)
