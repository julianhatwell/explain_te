import json
import time
import timeit
import numpy as np
from math import sqrt
from copy import deepcopy
import CHIRPS.datasets as ds
import CHIRPS.routines as rt
import CHIRPS.structures as strcts
from CHIRPS import p_count_corrected
from CHIRPS import config as cfg
from lime import lime_tabular as limtab
from anchor import anchor_tabular as anchtab
from defragTrees.defragTrees import DefragModel

penalise_bad_prediction = lambda mc, tc, value : value if mc == tc else 0 # for global interp methods

datasets = [
            ds.adult_small_samp,
            ds.bankmark_samp,
            ds.car,
            ds.cardio,
            ds.credit,
            ds.german,
            ds.lending_tiny_samp,
            ds.nursery_samp,
            ds.rcdv_samp
           ]

def export_data_splits(datasets, project_dir=None, random_state_splits=123):
    # general data preparation
    for dataset in datasets:
        mydata = dataset(project_dir=project_dir)

        # train test split - one off hard-coded random state.
        # later we will vary the state to generate different forests for e.g. benchmarking
        train_index, test_index = mydata.get_tt_split_idx(random_state=random_state_splits)
        # save to csv to use the same splits in methods that aren't available in Python
        mydata.tt_split(train_index, test_index).to_csv(mydata.get_save_path(),
                                                                encoded_features = mydata.features_enc)
    print('Exported train-test data for ' + str(len(datasets)) + ' datasets.')

def forest_prep(ds_container, meta_data,
                save_path=None, override_tuning=False,
                tuning_grid=None, identifier='main',
                plot_cm=False, plot_cm_norm=False):

    X_train=ds_container.X_train_enc
    X_test=ds_container.X_test_enc
    y_train=ds_container.y_train
    y_test=ds_container.y_test

    class_names=meta_data['class_names_label_order']
    random_state=meta_data['random_state']

    best_params, _ = rt.tune_rf(
     X=X_train,
     y=y_train,
     grid=tuning_grid,
     save_path=save_path,
     override_tuning=override_tuning,
     random_state=random_state)

    rf = rt.RandomForestClassifier(random_state=random_state, oob_score=True, **best_params)
    rf.fit(X_train, y_train)

    # the outputs of this function are:
    # cm - confusion matrix as 2d array
    # acc - accuracy of model = correctly classified instance / total number instances
    # coka - Cohen's kappa score. Accuracy adjusted for probability of correct by random guess. Useful for multiclass problems
    # prfs - precision, recall, f-score, support with the following signatures as a 2d array
    # 0 <= p, r, f <= 1. s = number of instances for each true class label (row sums of cm)
    rt.evaluate_model(y_true=y_test, y_pred=rf.predict(X_test),
                        class_names=class_names,
                        plot_cm=plot_cm, plot_cm_norm=plot_cm_norm, # False here will output the metrics and suppress the plots
                        save_path=save_path,
                        identifier=identifier,
                        random_state=random_state)

    return(rf)

def unseen_data_prep(ds_container, n_instances=1, batch_size=1, which_split='test'):
    # this will normalise the above parameters to the size of the dataset
    n_instances, n_batches = rt.batch_instance_ceiling(ds_container=ds_container, n_instances=n_instances, batch_size=batch_size)

    # this gets the next batch out of the data_split_container according to the required number of instances
    # all formats can be extracted, depending on the requirement
    # unencoded, encoded (sparse matrix is the type returned by scikit), ordinary dense matrix also available
    instances, instances_matrix, instances_enc, instances_enc_matrix, labels = ds_container.get_next(batch_size, which_split='test') # default

    return(instances, instances_matrix, instances_enc, instances_enc_matrix, labels)

# function to manage the whole run and evaluation
def CHIRPS_benchmark(forest, ds_container, meta_data,
                    batch_size=100, n_instances=100,
                    forest_walk_async=True,
                    chirps_explanation_async=True,
                    save_path='', save_sensitivity_path=None,
                    dataset_name='',
                    random_state=123, **kwargs):

    if save_sensitivity_path is None:
        save_sensitivity_path=save_path
    # 2. Prepare Unseen Data and Predictions

    print('Prepare Unseen Data and Predictions for CHIRPS benchmark')
    # OPTION 1 - batching (to be implemented in the new code, right now it will do just one batch)
    instances, _, instances_enc, instances_enc_matrix, labels = unseen_data_prep(ds_container,
                                                                                n_instances=n_instances,
                                                                                batch_size=batch_size)
    # get predictions
    preds = forest.predict(instances_enc)

    # 3.1 - Extract Tree Prediction Paths
    print('Walking forest for ' + str(len(labels)) + ' instances... (please wait)')

    # wrapper object needs the decision forest itself and the dataset meta data (we have a convenience function for this)
    f_walker = strcts.forest_walker(forest = forest, meta_data=meta_data)

    # set the timer
    eval_start_time = time.asctime( time.localtime(time.time()) )
    forest_walk_start_time = timeit.default_timer()

    # do the walk - returns a batch_paths_container (even for just one instance)
    # requires the X instances in a matrix (dense, ordinary numpy matrix) - this is available in the data_split_container
    bp_container = f_walker.forest_walk(instances = instances_enc_matrix
                            , labels = preds # we're explaining the prediction, not the true label!
                            , forest_walk_async = forest_walk_async)

    # stop the timer
    forest_walk_end_time = timeit.default_timer()
    forest_walk_elapsed_time = forest_walk_end_time - forest_walk_start_time
    forest_walk_mean_elapsed_time = forest_walk_elapsed_time/len(labels)

    print('Forest Walk with async = ' + str(forest_walk_async))
    print('Forest Walk time elapsed:', "{:0.4f}".format(forest_walk_elapsed_time), 'seconds')
    print()

    # 3.2-3.4 - Freqent pattern mining of paths, Score and sort mined path segments, Merge path segments into one rule
    # get what the model predicts on the training sample
    sample_labels = forest.predict(ds_container.X_train_enc)

    # build CHIRPS and a rule for each instance represented in the batch paths container
    CHIRPS = strcts.batch_CHIRPS_explainer(bp_container,
                                    forest=forest,
                                    sample_instances=ds_container.X_train_enc, # any representative sample can be used
                                    sample_labels=sample_labels,
                                    meta_data=meta_data,
                                    forest_walk_mean_elapsed_time=forest_walk_mean_elapsed_time)

    print('Running CHIRPS on a batch of ' + str(len(labels)) + ' instances... (please wait)')
    # start a timer
    ce_start_time = timeit.default_timer()

    CHIRPS.batch_run_CHIRPS(chirps_explanation_async=chirps_explanation_async, **kwargs)

    ce_end_time = timeit.default_timer()
    ce_elapsed_time = ce_end_time - ce_start_time
    print('CHIRPS time elapsed:', "{:0.4f}".format(ce_elapsed_time), 'seconds')
    print('CHIRPS with async = ' + str(chirps_explanation_async))
    print()

    # 4. Evaluating CHIRPS Explanations
    print('Evaluating found explanations')

    results_start_time = timeit.default_timer()

    # iterate over all the test instances (based on the ids in the index)
    # scoring will leave out the specific instance by this id.
    rt.evaluate_CHIRPS_explainers(CHIRPS, ds_container, labels.index,
                                  forest=forest,
                                  meta_data=meta_data,
                                  eval_start_time=eval_start_time,
                                  print_to_screen=False, # set True when running single instances
                                  save_results_path=save_sensitivity_path,
                                  dataset_name=dataset_name,
                                  save_results_file='CHIRPS' + '_rnst_' + str(random_state),
                                  save_CHIRPS=False)

    results_end_time = timeit.default_timer()
    results_elapsed_time = results_end_time - results_start_time
    print('CHIRPS batch results eval time elapsed:', "{:0.4f}".format(results_elapsed_time), 'seconds')
    # this completes the CHIRPS runs

def Anchors_preproc(ds_container, meta_data):

    # create the discretise function from LIME tabular
    disc = limtab.QuartileDiscretizer(np.array(ds_container.X_train),
                                      categorical_features=meta_data['categorical_features'],
                                      feature_names=meta_data['features'])

    # create a copy of the var_dict with updated labels
    var_dict_anch = deepcopy(meta_data['var_dict'])
    for vk in var_dict_anch.keys():
        if var_dict_anch[vk]['data_type'] == 'continuous':
            var_dict_anch[vk]['labels'] = disc.names[var_dict_anch[vk]['order_col']]
            var_dict_anch[vk]['data_type'] = 'discretised'

    var_dict_anch['categorical_names'] = {var_dict_anch[vk]['order_col'] : var_dict_anch[vk]['labels'] \
                                        for vk in var_dict_anch.keys() if not var_dict_anch[vk]['class_col']}

    # create discretised versions of the training and test data
    ds_container.X_train_matrix = disc.discretize(np.array(ds_container.X_train))
    ds_container.X_test_matrix = disc.discretize(np.array(ds_container.X_test))

        # fit the Anchors explainer. Onehot encode the data. Replace the data in the ds_container
    explainer = anchtab.AnchorTabularExplainer(meta_data['class_names'], meta_data['features'], ds_container.X_train_matrix, var_dict_anch['categorical_names'])
    explainer.fit(ds_container.X_train_matrix, ds_container.y_train, ds_container.X_test_matrix, ds_container.y_test)

    ds_container.X_train_enc = explainer.encoder.transform(ds_container.X_train_matrix)
    ds_container.X_test_enc = explainer.encoder.transform(ds_container.X_test_matrix)

    ds_container.X_train_enc_matrix = ds_container.X_train_enc.todense()
    ds_container.X_test_enc_matrix = ds_container.X_test_enc.todense()

    return(ds_container, explainer)

def Anchors_explanation(instance, explainer, forest, random_state=123, threshold=0.95):
    np.random.seed(random_state)
    exp = explainer.explain_instance(instance, forest.predict, threshold=threshold)
    return(exp)

# function to manage the whole run and evaluation
def Anchors_benchmark(forest, ds_container, meta_data,
                    anchors_explainer,
                    batch_size=100, n_instances=100,
                    save_path='',
                    dataset_name='',
                    precis_threshold=0.95,
                    random_state=123):

    identifier = 'Anchors'
    # 2. Prepare Unseen Data and Predictions
    print('Prepare Unseen Data and Predictions for Anchors benchmark')
    # OPTION 1 - batching (to be implemented in the new code, right now it will do just one batch)
    _, instances_matrix, instances_enc, _, labels = unseen_data_prep(ds_container,
                                            n_instances=n_instances,
                                            batch_size=batch_size)
    # get predictions
    preds = forest.predict(instances_enc)
    sample_labels = forest.predict(ds_container.X_train_enc) # for train estimates

    print('Running Anchors on each instance and collecting results')
    eval_start_time = time.asctime( time.localtime(time.time()) )
    # iterate through each instance to generate the anchors explanation
    results = [[]] * len(labels)
    evaluator = strcts.evaluator()
    for i in range(len(labels)):
        instance_id = labels.index[i]
        if i % 10 == 0: print('Working on ' + identifier + ' for instance ' + str(instance_id))

        # get test sample by leave-one-out on current instance
        _, loo_instances_matrix, loo_instances_enc, _, loo_true_labels = ds_container.get_loo_instances(instance_id, which_split='test')

        # get the model predicted labels
        loo_preds = forest.predict(loo_instances_enc)

        # collect the time taken
        anch_start_time = timeit.default_timer()

        # start the explanation process
        explanation = Anchors_explanation(instances_matrix[i], anchors_explainer, forest,
                                            threshold=precis_threshold,
                                            random_state=random_state)

        # the whole anchor explanation routine has run so stop the clock
        anch_end_time = timeit.default_timer()
        anch_elapsed_time = anch_end_time - anch_start_time

        # Get train and test idx (boolean) covered by the anchor
        anchor_train_idx = np.array([all_eq.all() for all_eq in ds_container.X_train_matrix[:, explanation.features()] == instances_matrix[:,explanation.features()][i]])
        anchor_test_idx = np.array([all_eq.all() for all_eq in loo_instances_matrix[:, explanation.features()] == instances_matrix[:,explanation.features()][i]])

        # create a class to run the standard evaluation
        train_metrics = evaluator.evaluate(prior_labels=sample_labels, post_idx=anchor_train_idx)
        test_metrics = evaluator.evaluate(prior_labels=loo_preds, post_idx=anchor_test_idx)

        # collect the results
        tc = [preds[i]]
        tc_lab = meta_data['get_label'](meta_data['class_col'], tc)

        results[i] = [dataset_name,
            instance_id,
            identifier,
            ' AND '.join(explanation.names()),
            len(explanation.names()),
            tc[0],
            tc_lab[0],
            tc[0],
            tc_lab[0],
            np.array([tree.predict(instances_enc[i]) == tc for tree in forest.estimators_]).mean(), # majority vote share
            test_metrics['prior']['p_counts'][tc][0],
            train_metrics['posterior'][tc][0],
            train_metrics['stability'][tc][0],
            train_metrics['recall'][tc][0],
            train_metrics['f1'][tc][0],
            train_metrics['accuracy'][tc][0],
            train_metrics['lift'][tc][0],
            train_metrics['coverage'],
            train_metrics['xcoverage'],
            train_metrics['kl_div'],
            test_metrics['posterior'][tc][0],
            test_metrics['stability'][tc][0],
            test_metrics['recall'][tc][0],
            test_metrics['f1'][tc][0],
            test_metrics['accuracy'][tc][0],
            test_metrics['lift'][tc][0],
            test_metrics['coverage'],
            test_metrics['xcoverage'],
            test_metrics['kl_div'],
            anch_elapsed_time]

    if save_path is not None:
        save_results_file = identifier + '_rnst_' + str(random_state)
        # save to file between each loop, in case of crashes/restarts
        rt.save_results(cfg.results_headers, results, save_results_path=save_path,
                        save_results_file=save_results_file)

        # collect summary_results
        with open(meta_data['get_save_path']() + 'forest_performance_rnst_' + str(meta_data['random_state']) + '.json', 'r') as infile:
            forest_performance = json.load(infile)
        f_perf = forest_performance['Anchors']['test_accuracy']
        p_perf = f_perf # for Anchors, forest pred and Anchors target are always the same
        fid = 1 # for Anchors, forest pred and Anchors target are always the same
        summary_results = [[dataset_name, identifier, len(labels), 1, \
                            1, 1, 1, 0, \
                            np.mean([rl_ln[4] for rl_ln in results]), np.std([rl_ln[4] for rl_ln in results]), \
                            eval_start_time, time.asctime( time.localtime(time.time()) ), \
                            f_perf, sqrt((f_perf/(1-f_perf))/len(labels)), \
                            1, 0, \
                            1, 1, 0]]

        rt.save_results(cfg.summary_results_headers, summary_results, save_path, save_results_file + '_summary')

def defragTrees_prep(forest, meta_data, ds_container,
                        Kmax=10, maxitr=100, restart=10,
                        identifier='defragTrees', save_path=''):

    X_train=ds_container.X_train_enc_matrix
    y_train=ds_container.y_train
    X_test=ds_container.X_test_enc_matrix
    y_test=ds_container.y_test

    feature_names = meta_data['features_enc']
    class_names = meta_data['class_names_label_order']
    random_state = meta_data['random_state']

    print('Running defragTrees')
    print()
    eval_start_time = time.asctime( time.localtime(time.time()) )
    defTrees_start_time = timeit.default_timer()

    # fit simplified model
    splitter = DefragModel.parseSLtrees(forest) # parse sklearn tree ensembles into the array of (feature index, threshold)
    mdl = DefragModel(modeltype='classification', maxitr=maxitr, qitr=0, tol=1e-6, restart=restart, verbose=0, seed=random_state)
    mdl.fit(np.array(X_train), y_train, splitter, Kmax, fittype='FAB', featurename=feature_names)

    defTrees_end_time = timeit.default_timer()
    defTrees_elapsed_time = defTrees_end_time - defTrees_start_time
    print('Fit defragTrees time elapsed:', "{:0.4f}".format(defTrees_elapsed_time), 'seconds')
    print()

    score, cover, coll = mdl.evaluate(np.array(X_test), y_test)
    print('defragTrees test accuracy')
    print('Accuracy = %f' % (1 - score,))
    print('Coverage = %f' % (cover,))
    print('Overlap = %f' % (coll,))

    return(mdl, eval_start_time, defTrees_elapsed_time)

def rule_list_from_dfrgtrs(dfrgtrs):
    rule_list = [[]] * len(dfrgtrs.rule_)
    for r, rule in enumerate(dfrgtrs.rule_):
        rule_list[r] = [(dfrgtrs.featurename_[int(item[0]-1)], not item[1], item[2]) for item in rule]

    return(rule_list)

def which_rule(rule_list, X, features):
    left_idx = np.array([i for i in range(len(X.todense()))])
    rules_idx = np.full(len(left_idx), np.nan)
    rule_cascade = 0
    while len(left_idx) > 0 and rule_cascade < len(rule_list):
        rule_evaluator = strcts.rule_evaluator()
        match_idx = rule_evaluator.apply_rule(rule=rule_list[rule_cascade],
                                              instances=X[left_idx,:],
                                              features=features)
        rules_idx[left_idx[match_idx]] = rule_cascade
        left_idx = np.array([li for li, mi in zip(left_idx, match_idx) if not mi])
        rule_cascade += 1
    # what's left is the default prediction
    rules_idx[np.where(np.isnan(rules_idx))] = rule_cascade # default prediction
    rules_idx = rules_idx.astype(int)
    return(rules_idx)

def defragTrees_benchmark(forest, ds_container, meta_data, dfrgtrs,
                            eval_start_time, defTrees_elapsed_time,
                            batch_size=100, n_instances=100,
                            save_path='', dataset_name='',
                            random_state=123):

    identifier = 'defragTrees'
    print('defragTrees benchmark')
    # OPTION 1 - batching (to be implemented in the new code, right now it will do just one batch)
    _, _, instances_enc, instances_enc_matrix, labels = unseen_data_prep(ds_container,
                                                                            n_instances=n_instances,
                                                                            batch_size=batch_size)

    defTrees_mean_elapsed_time = defTrees_elapsed_time / len(labels)

    eval_model = rt.evaluate_model(y_true=labels, y_pred=dfrgtrs.predict(np.array(instances_enc_matrix)),
                        class_names=meta_data['class_names_label_order'],
                        plot_cm=False, plot_cm_norm=False, # False here will output the metrics and suppress the plots
                        save_path=save_path,
                        identifier=identifier,
                        random_state=random_state)

    forest_preds = forest.predict(instances_enc)
    dfrgtrs_preds = dfrgtrs.predict(np.array(instances_enc_matrix))

    rule_list = rule_list_from_dfrgtrs(dfrgtrs)

    results = [[]] * len(labels)
    rule_idx = []
    evaluator = strcts.evaluator()
    for i in range(len(labels)):
        instance_id = labels.index[i]
        if i % 10 == 0: print('Working on ' + identifier + ' for instance ' + str(instance_id))

        # get test sample by leave-one-out on current instance
        _, _, loo_instances_enc, loo_instances_enc_matrix, loo_true_labels = ds_container.get_loo_instances(instance_id,
                                                                                                            which_split='test')

        loo_forest_preds = forest.predict(loo_instances_enc)
        loo_dfrgtrs_preds = dfrgtrs.predict(np.array(loo_instances_enc_matrix))

        # start a timer for the individual eval
        dt_start_time = timeit.default_timer()

        # which rule appies to each loo instance
        rule_idx.append(which_rule(rule_list, loo_instances_enc, features=meta_data['features_enc']))

        # which rule appies to current instance
        rule = which_rule(rule_list, instances_enc[i], features=meta_data['features_enc'])
        if rule[0] >= len(rule_list):
            pretty_rule = []
        else:
            pretty_rule = evaluator.prettify_rule(rule_list[rule[0]], meta_data['var_dict'])

        dt_end_time = timeit.default_timer()
        dt_elapsed_time = dt_end_time - dt_start_time
        dt_elapsed_time = dt_elapsed_time + defTrees_mean_elapsed_time # add the mean modeling time per instance

        # which instances are covered by this rule
        covered_instances = rule_idx[i] == rule

        metrics = evaluator.evaluate(prior_labels=loo_true_labels,
                                        post_idx=covered_instances)

        # majority class is the forest vote class
        # target class is the benchmark algorithm prediction
        mc = [forest_preds[i]]
        tc = [dfrgtrs_preds[i]]
        mc_lab = meta_data['get_label'](meta_data['class_col'], mc)
        tc_lab = meta_data['get_label'](meta_data['class_col'], tc)

        results[i] = [dataset_name,
        instance_id,
        identifier,
        pretty_rule,
        len(rule),
        mc[0],
        mc_lab[0],
        tc[0],
        tc_lab[0],
        np.array([tree.predict(instances_enc[i]) == mc for tree in forest.estimators_]).mean(), # majority vote share
        metrics['prior']['p_counts'][mc][0],
        metrics['posterior'][tc][0],
        metrics['stability'][tc][0],
        metrics['recall'][tc][0],
        metrics['f1'][tc][0],
        metrics['accuracy'][tc][0],
        metrics['lift'][tc][0],
        metrics['coverage'],
        metrics['xcoverage'],
        metrics['kl_div'],
        penalise_bad_prediction(mc, tc, metrics['posterior'][mc][0]),
        penalise_bad_prediction(mc, tc, metrics['stability'][mc][0]),
        penalise_bad_prediction(mc, tc, metrics['recall'][mc][0]),
        penalise_bad_prediction(mc, tc, metrics['f1'][mc][0]),
        penalise_bad_prediction(mc, tc, metrics['accuracy'][mc][0]),
        penalise_bad_prediction(mc, tc, metrics['lift'][mc][0]),
        metrics['coverage'],
        metrics['xcoverage'],
        metrics['kl_div'],
        dt_elapsed_time]

    if save_path is not None:
        save_results_file=identifier + '_rnst_' + str(random_state)

        rt.save_results(cfg.results_headers, results,
                        save_results_path=save_path,
                        save_results_file=save_results_file)

        # collect summary_results
        with open(meta_data['get_save_path']() + 'forest_performance_rnst_' + str(meta_data['random_state']) + '.json', 'r') as infile:
            forest_performance = json.load(infile)
        f_perf = forest_performance['main']['test_accuracy']
        p_perf = np.mean(dfrgtrs_preds == labels)
        fid = np.mean(dfrgtrs_preds == forest_preds)
        summary_results = [[dataset_name, identifier, len(labels), len(rule_list), \
                            len(np.unique(rule_idx)), np.median(np.array(rule_idx) + 1), np.mean(np.array(rule_idx) + 1), np.std(np.array(rule_idx) + 1), \
                            np.mean([rl_ln[4] for rl_ln in results]), np.std([rl_ln[4] for rl_ln in results]), \
                            eval_start_time, time.asctime( time.localtime(time.time()) ), \
                            f_perf, sqrt((f_perf/(1-f_perf))/len(labels)), \
                            p_perf, sqrt((p_perf/(1-p_perf))/len(labels)), \
                            eval_model['test_kappa'], fid, sqrt((fid/(1-fid))/len(labels))]]

        rt.save_results(cfg.summary_results_headers, summary_results,
                        save_results_path=save_path,
                        save_results_file=save_results_file + '_summary')
