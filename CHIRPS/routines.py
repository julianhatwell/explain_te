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

from CHIRPS import if_nexists_make_dir, if_nexists_make_file, chisq_indep_test, p_count_corrected
from CHIRPS.plotting import plot_confusion_matrix

from CHIRPS import config as cfg

# bug in sk-learn. Should be fixed in August
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

def extend_path(stem, extensions, is_dir=False):
    # add the extension and the path separator
    for x in extensions:
        stem = stem + x + cfg.path_sep
    # just add the final extension
    if is_dir:
        return(stem)
    else:
        return(stem[:-1])

def do_tuning(X, y, grid = None, random_state=123, save_path = None):
    if grid is None:
        grid = ParameterGrid({
            'n_estimators': [(i + 1) * 200 for i in range(8)]
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

    params = DataFrame(params).sort_values(['score','n_estimators' ],
                                        ascending=[False, True])

    best_grid = params.loc[params['score'].idxmax()]
    best_params = {k: int(v) for k, v in best_grid.items() if k != 'score'}
    forest_performance = {'oobe_score' : best_grid['score'],
                        'fitting_time' : fitting_elapsed_time}

    if save_path is not None:
        if_nexists_make_dir(save_path)
        with open(save_path + 'best_params_rnst_' + str(random_state) + '.json', 'w') as outfile:
            json.dump(best_params, outfile)
        with open(save_path + 'forest_performance_rnst_' + str(random_state) + '.json', 'w') as outfile:
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
            with open(save_path + 'best_params_rnst_' + str(random_state) + '.json', 'r') as infile:
                print('using previous tuning parameters')
                best_params = json.load(infile)
            with open(save_path + 'forest_performance_rnst_' + str(random_state) + '.json', 'r') as infile:
                forest_performance = json.load(infile)
            return(best_params, forest_performance)
        except:
            print('New grid tuning... (please wait)')
            best_params, forest_performance = do_tuning(X, y, grid=grid, random_state=random_state, save_path=save_path)

    print('Best OOB Accuracy Estimate during tuning: ' '{:0.4f}'.format(forest_performance['oobe_score']))
    print('Best parameters:', best_params)
    print()

    return(best_params, forest_performance)

def update_model_performance(save_path, test_metrics, identifier, random_state):
    with open(save_path + 'forest_performance_rnst_' + str(random_state) + '.json', 'r') as infile:
        forest_performance = json.load(infile)
    forest_performance.update({identifier : test_metrics})
    with open(save_path + 'forest_performance_rnst_' + str(random_state) + '.json', 'w') as outfile:
        json.dump(forest_performance, outfile)

def evaluate_model(y_true, y_pred, class_names=None,
                    print_metrics=False, plot_cm=True, plot_cm_norm=True,
                    save_path=None, identifier = 'main', random_state=123):

    # view the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    prfs = precision_recall_fscore_support(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    coka = cohen_kappa_score(y_true, y_pred)

    test_metrics = {'confmat' : cm.tolist(),
                            'test_accuracy' : acc,
                            'test_kappa' : coka,
                            'test_prec' : prfs[0].tolist(),
                            'test_recall' : prfs[1].tolist(),
                            'test_f1' : prfs[2].tolist(),
                            'test_prior' : (prfs[3] / prfs[3].sum()).tolist(), # true labels. confmat rowsums
                            'test_posterior' : (cm.sum(axis=0) / prfs[3].sum()).tolist() } # pred labels. confmat colsums
    if print_metrics:
        print(test_metrics)

    if save_path is not None:
        update_model_performance(save_path, test_metrics, identifier, random_state)


    if plot_cm:
        plot_confusion_matrix(cm, class_names=class_names,
                              title='Confusion matrix, without normalization')
    # normalized confusion matrix
    if plot_cm_norm:
        plot_confusion_matrix(cm
                              , class_names=class_names
                              , normalize=True,
                              title='Confusion matrix normalized on rows (predicted label share)')

    return(test_metrics)

def batch_instance_ceiling(ds_container, n_instances=None, batch_size=None):
    dataset_size = len(ds_container.y_test)
    if batch_size is None:
        batch_size = dataset_size
    if n_instances is None:
        n_instances = dataset_size
    n_instances = min(n_instances, dataset_size)
    batch_size = min(batch_size, dataset_size)
    n_batches = int(batch_size / n_instances)
    return(n_instances, n_batches)

def save_results(results, save_results_path, save_results_file):
    # create new directory if necessary
    if_nexists_make_dir(save_results_path)
    # save the tabular results to a file
    headers = ['dataset_name', 'instance_id', 'algorithm',
                'pretty rule', 'rule length',
                'pred class', 'pred class label',
                'target class', 'target class label',
                'forest vote share', 'pred prior',
                'precision(tr)', 'stability(tr)', 'recall(tr)',
                'f1(tr)', 'accuracy(tr)', 'lift(tr)',
                'coverage(tr)', 'xcoverage(tr)', 'kl_div(tr)',
                'precision(tt)', 'stability(tt)', 'recall(tt)',
                'f1(tt)', 'accuracy(tt)', 'lift(tt)',
                'coverage(tt)', 'xcoverage(tt)', 'kl_div(tt)']
    output_df = DataFrame(results, columns=headers)
    output_df.to_csv(save_results_path + save_results_file + '.csv')

def evaluate_CHIRPS_explainers(b_CHIRPS_exp, # batch_CHIRPS_explainer
                                ds_container, # data_split_container (for the test data and the LOO function
                                instance_idx, # should match the instances in the batch
                                forest,
                                dataset_name='',
                                eval_alt_labelings=False,
                                eval_rule_complements=False,
                                print_to_screen=False,
                                save_results_path=None,
                                save_results_file=None,
                                save_CHIRPS=False):

    results = [[]] * len(b_CHIRPS_exp.CHIRPS_explainers)

    for i, c in enumerate(b_CHIRPS_exp.CHIRPS_explainers):

        # get test sample by leave-one-out on current instance
        instance_id = instance_idx[i]
        _, _, instances_enc, _, true_labels = ds_container.get_loo_instances(instance_id)
        # get the model predicted labels
        labels = Series(forest.predict(instances_enc), index = true_labels.index)

        # get the detail of the current index
        _, _, current_instance_enc, _, current_instance_label = ds_container.get_by_id([instance_id], which_split='test')

        # then evaluating rule metrics on the leave-one-out test set
        eval_rule = c.evaluate_rule(rule='pruned', sample_instances=instances_enc, sample_labels=labels)
        tc = c.target_class
        tc_lab = c.target_class_label

        # collect results
        tt_prior = (labels.value_counts() / len(labels)).values
        tt_prior_counts = eval_rule['prior']['counts']
        tt_posterior = eval_rule['posterior']
        tt_posterior_counts = eval_rule['counts']
        tt_chisq = chisq_indep_test(tt_posterior_counts, tt_prior_counts)[1]
        tt_prec = eval_rule['posterior'][tc]
        tt_stab = eval_rule['stability'][tc]
        tt_recall = eval_rule['recall'][tc]
        tt_f1 = eval_rule['f1'][tc]
        tt_acc = eval_rule['accuracy'][tc]
        tt_lift = eval_rule['lift'][tc]
        tt_coverage = eval_rule['coverage']
        tt_xcoverage = eval_rule['xcoverage']
        tt_kl_div = eval_rule['kl_div']

        if eval_rule_complements:
            rule_complement_results = c.eval_rule_complements(sample_instances=instances_enc, sample_labels=labels)

        if eval_alt_labelings:
            # get the current instance being explained
            # get_by_id takes a list of instance ids. Here we have just a single integer
            alt_labelings_results = c.get_alt_labelings(instance=current_instance_enc,
                                                        sample_instances=instances_enc,
                                                        forest=forest)

        results[i] = [dataset_name,
            instance_id,
            c.algorithm,
            c.pretty_rule,
            c.rule_len,
            c.major_class,
            c.major_class_label,
            c.target_class,
            c.target_class_label,
            c.forest_vote_share,
            c.prior[tc],
            c.est_prec,
            c.est_stab,
            c.est_recall,
            c.est_f1,
            c.est_acc,
            c.est_lift,
            c.est_coverage,
            c.est_xcoverage,
            c.est_kl_div,
            tt_prec,
            tt_stab,
            tt_recall,
            tt_f1,
            tt_acc,
            tt_lift,
            tt_coverage,
            tt_xcoverage,
            tt_kl_div]

        if print_to_screen:
            print('INSTANCE RESULTS')
            print('instance id: ' + str(instance_id) + ' with true class label: ' + str(current_instance_label.values[0]) + \
                    ' (' + c.get_label(c.class_col, current_instance_label.values[0]) + ')')
            print()
            c.to_screen()
            print('Results - Previously Unseen Sample')
            print('target class prior (unseen data): ' + str(tt_prior[tc]))
            print('rule coverage (unseen data): ' + str(tt_coverage))
            print('rule xcoverage (unseen data): ' + str(tt_xcoverage))
            print('rule precision (unseen data): ' + str(tt_prec))
            print('rule stability (unseen data): ' + str(tt_stab))
            print('rule recall (unseen data): ' + str(tt_recall))
            print('rule f1 score (unseen data): ' + str(tt_f1))
            print('rule lift (unseen data): ' + str(tt_lift))
            print('prior (unseen data): ' + str(tt_prior))
            print('prior counts (unseen data): ' + str(tt_prior_counts))
            print('rule posterior (unseen data): ' + str(tt_posterior))
            print('rule posterior counts (unseen data): ' + str(tt_posterior_counts))
            print('rule chisq p-value (unseen data): ' + str(tt_chisq))
            print('rule Kullback-Leibler divergence (unseen data): ' + str(tt_kl_div))
            print()
            if eval_rule_complements:
                print('RULE COMPLEMENT RESULTS')
                for rcr in rule_complement_results:
                    eval_rule = rcr['eval']
                    tt_posterior = eval_rule['posterior']
                    tt_posterior_counts = eval_rule['counts']
                    tt_chisq = chisq_indep_test(tt_posterior_counts, tt_prior_counts)[1]
                    tt_prec = eval_rule['posterior'][tc]
                    tt_stab = eval_rule['stability'][tc]
                    tt_recall = eval_rule['recall'][tc]
                    tt_f1 = eval_rule['f1'][tc]
                    tt_acc = eval_rule['accuracy'][tc]
                    tt_lift = eval_rule['lift'][tc]
                    tt_coverage = eval_rule['coverage']
                    tt_xcoverage = eval_rule['xcoverage']
                    kl_div = rcr['kl_div']
                    print('Feature Reversed: ' + rcr['feature'])
                    print('rule: ' + rcr['pretty_rule'])
                    print('rule coverage (unseen data): ' + str(tt_coverage))
                    print('rule xcoverage (unseen data): ' + str(tt_xcoverage))
                    print('rule precision (unseen data): ' + str(tt_prec))
                    print('rule stability (unseen data): ' + str(tt_stab))
                    print('rule recall (unseen data): ' + str(tt_recall))
                    print('rule f1 score (unseen data): ' + str(tt_f1))
                    print('rule lift (unseen data): ' + str(tt_lift))
                    print('prior (unseen data): ' + str(tt_prior))
                    print('prior counts (unseen data): ' + str(tt_prior_counts))
                    print('rule posterior (unseen data): ' + str(tt_posterior))
                    print('rule posterior counts (unseen data): ' + str(tt_posterior_counts))
                    print('rule chisq p-value (unseen data): ' + str(tt_chisq))
                    print('rule Kullback-Leibler divergence from original: ' + str(kl_div))
                    if eval_alt_labelings:
                        for alt_labels in alt_labelings_results:
                            if alt_labels['feature'] == rcr['feature']:
                                print('predictions for this rule complement')
                                if not alt_labels['mask_cover']:
                                    print('note: this combination does not exist in the original data \
                                    \nexercise caution when interpreting the results.')
                                print('instance specific. expected class: ' + str(np.argmax(alt_labels['is_mask']['p_counts'])) + \
                                        ' (' + c.get_label(c.class_col, np.argmax(alt_labels['is_mask']['p_counts'])) + ')')
                                print('classes: ' + str(alt_labels['is_mask']['labels']))
                                print('counts: ' + str(alt_labels['is_mask']['counts']))
                                print('proba: ' + str(alt_labels['is_mask']['p_counts']))
                                print('allowed values. expected class: ' + str(np.argmax(alt_labels['av_mask']['p_counts'])) + \
                                        ' (' + c.get_label(c.class_col, np.argmax(alt_labels['is_mask']['p_counts'])) + ')')
                                print('classes: ' + str(alt_labels['av_mask']['labels']))
                                print('counts: ' + str(alt_labels['av_mask']['counts']))
                                print('proba: ' + str(alt_labels['av_mask']['p_counts']))
                                print()
                    else:
                        print()

    if save_results_path is not None:
        save_results(results, save_results_path, save_results_file)

    if save_CHIRPS:
        # save the batch_CHIRPS_explainer object
        CHIRPS_explainers_store = open(save_results_path + save_results_file + '.pickle', "wb")
        pickle.dump(b_CHIRPS_exp.CHIRPS_explainers, CHIRPS_explainers_store)
        CHIRPS_explainers_store.close()
