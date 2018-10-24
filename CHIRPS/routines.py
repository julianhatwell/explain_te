import json
import time
import timeit
import pickle
import numpy as np
import multiprocessing as mp
from pandas import DataFrame, Series

from scipy.stats import chi2_contingency
from math import sqrt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_recall_fscore_support, accuracy_score

from CHIRPS import if_nexists_make_dir, chisq_indep_test
from CHIRPS.plotting import plot_confusion_matrix

# bug in sk-learn. Should be fixed in August
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

def do_tuning(X, y, grid = None, random_state=123, save_path = None):
    if grid is None:
        grid = ParameterGrid({
            'n_estimators': [(i + 1) * 500 for i in range(3)]
            , 'max_depth' : [i for i in [8, 16]]
            , 'min_samples_leaf' : [1, 5]
            })

    start_time = timeit.default_timer()

    rf = RandomForestClassifier()
    params = []
    best_score = 0

    for g in grid:
        print('Trying params: ' + str(g))
        fitting_start_time = timeit.default_timer()
        rf.set_params(oob_score = True, random_state=random_state, **g)
        rf.fit(X, y)
        fitting_end_time = timeit.default_timer()
        fitting_elapsed_time = fitting_end_time - fitting_start_time
        oobe = rf.oob_score_
        print('Training time: ' + str(fitting_elapsed_time))
        print('Out of Bag Accuracy Score: ' + str(oobe))
        print()
        g['score'] = oobe
        params.append(g)

    elapsed = timeit.default_timer() - start_time

    params = DataFrame(params).sort_values(['score','n_estimators','max_depth','min_samples_leaf'],
                                        ascending=[False, True, True, False])

    best_grid = params.loc[params['score'].idxmax()]
    best_params = {k: int(v) for k, v in best_grid.items() if k != 'score'}
    forest_performance = {'oobe_score' : best_grid['score'],
                        'fitting_time' : fitting_elapsed_time}

    if save_path is not None:
        if_nexists_make_dir(save_path)
        with open(save_path + 'best_params_rndst_' + str(random_state) + '.json', 'w') as outfile:
            json.dump(best_params, outfile)
        with open(save_path + 'forest_performance_rndst_' + str(random_state) + '.json', 'w') as outfile:
            json.dump(forest_performance, outfile)

    return(best_params, forest_performance)

def tune_rf(X, y, grid = None, random_state=123, save_path = None, override_tuning=False):

    # to do - test allowable structure of grid input
    if override_tuning:
        print('Over-riding previous tuning parameters. New grid tuning... (please wait)')
        print()
        tun_start_time = timeit.default_timer()
        best_params, forest_performance = do_tuning(X, y, grid=grid, random_state=random_state, save_path=save_path)
        tun_elapsed_time = timeit.default_timer() - tun_start_time
        print('Tuning time elapsed:', "{:0.4f}".format(tun_elapsed_time), 'seconds')
    else:
        try:
            with open(save_path + 'best_params_rndst_' + str(random_state) + '.json', 'r') as infile:
                print('using previous tuning parameters')
                best_params = json.load(infile)
            with open(save_path + 'forest_performance_rndst_' + str(random_state) + '.json', 'r') as infile:
                forest_performance = json.load(infile)
            return(best_params, forest_performance)
        except:
            print('New grid tuning... (please wait)')
            best_params, forest_performance = do_tuning(X, y, grid=grid, random_state=random_state, save_path=save_path)

    print('Best OOB Accuracy Estimate during tuning: ' '{:0.4f}'.format(forest_performance['oobe_score']))
    print('Best parameters:', best_params)
    print()

    return(best_params, forest_performance)

def evaluate_model(X, y, prediction_model, class_names=None, plot_cm=True, plot_cm_norm=True):
    pred = prediction_model.predict(X)

    # view the confusion matrix
    cm = confusion_matrix(y, pred)
    prfs = precision_recall_fscore_support(y, pred)
    acc = accuracy_score(y, pred)
    coka = cohen_kappa_score(y, pred)

    if plot_cm:
        plot_confusion_matrix(cm, class_names=class_names,
                              title='Confusion matrix, without normalization')
    # normalized confusion matrix
    if plot_cm_norm:
        plot_confusion_matrix(cm
                              , class_names=class_names
                              , normalize=True,
                              title='Confusion matrix normalized on rows (predicted label share)')
    return(cm, acc, coka, prfs)

def update_model_performance(save_path, random_state, test_metrics):
    with open(save_path + 'forest_performance_rndst_' + str(random_state) + '.json', 'r') as infile:
        forest_performance = json.load(infile)
    forest_performance.update(test_metrics)
    with open(save_path + 'forest_performance_rndst_' + str(random_state) + '.json', 'w') as outfile:
        json.dump(forest_performance, outfile)

def batch_instance_ceiling(data_split, n_instances=None, batch_size=None):
    dataset_size = len(data_split.y_test)
    if batch_size is None:
        batch_size = dataset_size
    if n_instances is None:
        n_instances = dataset_size
    n_instances = min(n_instances, dataset_size)
    batch_size = min(batch_size, dataset_size)
    n_batches = int(batch_size / n_instances)
    return(n_instances, n_batches)

def evaluate_CHIRPS_explainers(b_CHIRPS_exp, # batch_CHIRPS_explainer
                                ds_container, # data_split_container (for the test data and the LOO function
                                instance_idx, # should match the instances in the batch
                                eval_alt_labelings=False,
                                eval_rule_complements=False,
                                print_to_screen=False,
                                save_results_path=None,
                                save_results_file=None,
                                save_CHIRPS=False):

    output = [[]] * len(b_CHIRPS_exp.CHIRPS_explainers)

    for i, c in enumerate(b_CHIRPS_exp.CHIRPS_explainers):

        # get test sample by leave-one-out on current instance
        instance_id = instance_idx[i]
        _, instances_enc, _, labels = ds_container.get_loo_instances(instance_id)

        # then evaluating rule metrics on the leave-one-out test set
        eval_rule = c.evaluate_rule(rule='pruned', instances=instances_enc, labels=labels)
        tc = c.target_class
        tc_lab = c.target_class_label

        # collect results
        tt_prior = labels.value_counts()[0] / len(labels)
        tt_prior_counts = eval_rule['priors']['counts']
        tt_posterior_counts = eval_rule['counts']
        tt_chisq = chisq_indep_test(tt_posterior_counts, tt_prior_counts)[1]
        tt_prec = eval_rule['post'][tc]
        tt_stab = eval_rule['stability'][tc]
        tt_recall = eval_rule['recall'][tc]
        tt_f1 = eval_rule['f1'][tc]
        tt_acc = eval_rule['accuracy'][tc]
        tt_lift = eval_rule['lift'][tc]
        tt_coverage = eval_rule['coverage']
        tt_xcoverage = eval_rule['xcoverage']

        rule_complements = c.get_rule_complements() # get them anyway as they can be used in two optional places
        if eval_rule_complements:
            rule_complement_results = []
            for rc in rule_complements:
                rule_complement_results.append({'rule' : rc,
                                                'pretty_rule' : c.prettify_rule(rc),
                                                'eval' : c.evaluate_rule(rule=rc, instances=instances_enc, labels=labels)})

            print(rule_complement_results)
            print()

        if eval_alt_labelings:
            # get the current instance being explained
            # get_by_id takes a list of instance ids. Here we have just a single integer
            _, instances_enc, _, _ = ds_container.get_by_id([instance_id], which_split='test')
            # for each rc, create a dataset of the same size as what we are testing
            # the set is the instance that we are testing identical in every way
            # we will then replace one column with what the not rule says and see what happens

        output[i] = [instance_id,
            c.algorithm,
            c.pretty_rule,
            c.rule_len,
            c.major_class,
            c.major_class_label,
            c.target_class,
            c.target_class_label,
            c.forest_vote_share,
            c.prior,
            c.est_prec,
            c.est_stab,
            c.est_recall,
            c.est_f1,
            c.est_acc,
            c.est_lift,
            c.est_coverage,
            c.est_xcoverage,
            tt_prec,
            tt_stab,
            tt_recall,
            tt_f1,
            tt_acc,
            tt_lift,
            tt_coverage,
            tt_xcoverage]

        if print_to_screen:
            print('INSTANCE RESULTS')
            print('instance id: ' + str(instance_id) + ' with target class ' + str(tc) + ' (' + tc_lab + ')')
            print('target class prior (training data): ' + str(tt_prior))
            print()
            c.to_screen()
            print('Results - Previously Unseen Sample')
            print('rule coverage (unseen data): ' + str(tt_coverage))
            print('rule xcoverage (unseen data): ' + str(tt_xcoverage))
            print('rule precision (unseen data): ' + str(tt_prec))
            print('rule stability (unseen data): ' + str(tt_stab))
            print('rule recall (unseen data): ' + str(tt_recall))
            print('rule f1 score (unseen data): ' + str(tt_f1))
            print('rule lift (unseen data): ' + str(tt_lift))
            print('prior counts (unseen data): ' + str(tt_prior_counts))
            print('rule posterior counts (unseen data): ' + str(tt_posterior_counts))
            print('rule chisq p-value (unseen data): ' + str(tt_chisq))
            print()
            print()

    if save_results_path is not None:
        # create new directory if necessary
        if_nexists_make_dir(save_results_path)
        # save the tabular results to a file
        headers = ['instance_id', 'algorithm',
                    'pretty rule', 'rule length',
                    'pred class', 'pred class label',
                    'target class', 'target class label',
                    'forest vote share', 'pred prior',
                    'precision(tr)', 'stability(tr)', 'recall(tr)',
                    'f1(tr)', 'accuracy(tr)', 'lift(tr)',
                    'coverage(tr)', 'xcoverage(tr)',
                    'precision(tt)', 'stability(tt)', 'recall(tt)',
                    'f1(tt)', 'accuracy(tt)', 'lift(tt)',
                    'coverage(tt)', 'xcoverage(tt)']
        output_df = DataFrame(output, columns=headers)
        output_df.to_csv(save_results_path + save_results_file + '.csv')

        if save_CHIRPS:
            # save the batch_CHIRPS_explainer object
            CHIRPS_explainers_store = open(save_results_path + save_results_file + '.pickle', "wb")
            pickle.dump(b_CHIRPS_exp.CHIRPS_explainers, CHIRPS_explainers_store)
            CHIRPS_explainers_store.close()

def anchors_preproc(dataset, random_state, iv_low, iv_high):
    mydata = dataset(random_state)
    tt = mydata.xval_split(iv_low=iv_low, iv_high=iv_high, test_index=random_state, random_state=123)

    # mappings for anchors
    mydata.class_names=mydata.get_label(mydata.class_col, [i for i in range(len(mydata.class_names))]).tolist()
    mydata.unsorted_categorical = [(v, mydata.var_dict[v]['order_col']) for v in mydata.var_dict if mydata.var_dict[v]['data_type'] == 'nominal' and mydata.var_dict[v]['class_col'] != True]
    mydata.categorical_features = [c[1] for c in sorted(mydata.unsorted_categorical, key = lambda x: x[1])]
    mydata.categorical_names = {i : mydata.var_dict[v]['labels'] for v, i in mydata.unsorted_categorical}

    # discretizes all cont vars
    disc = limtab.QuartileDiscretizer(data=np.array(mydata.data.drop(labels=mydata.class_col, axis=1)),
                                             categorical_features=mydata.categorical_features,
                                             feature_names=mydata.features)

    # update the tt object
    tt['X_train'] = np.array(disc.discretize(np.array(tt['X_train'])))
    tt['X_test'] = np.array(disc.discretize(np.array(tt['X_test'])))
    tt['y_train'] = np.array(tt['y_train'])
    tt['y_test'] = np.array(tt['y_test'])

    # add the mappings of discretized vars for anchors
    mydata.categorical_names.update(disc.names)

    explainer = anchtab.AnchorTabularExplainer(mydata.class_names, mydata.features, tt['X_train'], mydata.categorical_names)
    explainer.fit(tt['X_train'], tt['y_train'], tt['X_test'], tt['y_test'])
    # update the tt object
    tt['encoder'] = explainer.encoder
    tt['X_train_enc'] = explainer.encoder.transform(tt['X_train'])

    return(mydata, tt, explainer)

def anchors_explanation(instance, explainer, forest, random_state=123, threshold=0.95):
    np.random.seed(random_state)
    exp = explainer.explain_instance(instance, forest.predict, threshold=threshold)
    return(exp)
