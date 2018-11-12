import json
import time
import timeit
import numpy as np
from pandas import Series
from copy import deepcopy
import CHIRPS.datasets as ds
import CHIRPS.routines as rt
import CHIRPS.structures as strcts
from CHIRPS import p_count_corrected
from lime import lime_tabular as limtab
from anchor import anchor_tabular as anchtab
from defragTrees.defragTrees import DefragModel

datasets = [
            ds.adult_small_samp_data,
            ds.bankmark_samp_data,
            ds.car_data,
            ds.cardio_data,
            ds.credit_data,
            ds.german_data,
            ds.lending_tiny_samp_data,
            ds.nursery_samp_data,
            # ds.rcdv_samp_data
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

def forest_prep(X_train, y_train, X_test, y_test, meta_data,
                save_path=None, override_tuning=False,
                tuning_grid=None, identifier='main'):

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
                        plot_cm=False, plot_cm_norm=False, # False here will output the metrics and suppress the plots
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
                    save_path='', dataset_name='',
                    random_state=123):
    # 2. Prepare Unseen Data and Predictions
    print('Prepare Unseen Data and Predictions for CHIRPS benchmark')
    # OPTION 1 - batching (to be implemented in the new code, right now it will do just one batch)
    instances, _, instances_enc, instances_enc_matrix, labels = unseen_data_prep(ds_container,
                                                                                n_instances=n_instances,
                                                                                batch_size=batch_size)
    # get predictions
    preds = Series(forest.predict(instances_enc), index = labels.index)

    # 3.1 - Extract Tree Prediction Paths
    print('Walking forest for ' + str(len(labels)) + ' instances... (please wait)')

    # wrapper object needs the decision forest itself and the dataset meta data (we have a convenience function for this)
    f_walker = strcts.forest_walker(forest = forest, meta_data=meta_data)

    # set the timer
    forest_walk_start_time = timeit.default_timer()

    # do the walk - returns a batch_paths_container (even for just one instance)
    # requires the X instances in a matrix (dense, ordinary numpy matrix) - this is available in the data_split_container
    bp_container = f_walker.forest_walk(instances = instances_enc_matrix
                            , labels = preds.values # we're explaining the prediction, not the true label!
                            , forest_walk_async = forest_walk_async)

    # stop the timer
    forest_walk_end_time = timeit.default_timer()
    forest_walk_elapsed_time = forest_walk_end_time - forest_walk_start_time

    print('Forest Walk with async = ' + str(forest_walk_async))
    print('Forest Walk time elapsed:', "{:0.4f}".format(forest_walk_elapsed_time), 'seconds')
    print()

    # 3.2-3.4 - Freqent pattern mining of paths, Score and sort mined path segments, Merge path segments into one rule
    # get what the model predicts on the training sample
    sample_labels = Series(forest.predict(ds_container.X_train_enc), index = ds_container.y_train.index)

    # build CHIRPS and a rule for each instance represented in the batch paths container
    CHIRPS = strcts.batch_CHIRPS_explainer(bp_container,
                                    forest=forest,
                                    sample_instances=ds_container.X_train_enc, # any representative sample can be used
                                    # sample_labels=tt.y_train,  # any representative sample can be used
                                    sample_labels=sample_labels,
                                    meta_data=meta_data)

    print('Running CHIRPS on a batch of ' + str(len(labels)) + ' instances... (please wait)')
    # start a timer
    ce_start_time = timeit.default_timer()

    CHIRPS.batch_run_CHIRPS(chirps_explanation_async=chirps_explanation_async) # all the defaults

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
                                  print_to_screen=False, # set True when running single instances
                                  save_results_path=save_path,
                                  dataset_name=dataset_name,
                                  save_results_file='CHIRPS_results' + '_rnst_' + str(random_state),
                                  save_CHIRPS=True)

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
    _, _, _, _, labels = unseen_data_prep(ds_container,
                                            n_instances=n_instances,
                                            batch_size=batch_size)
    # get predictions
    preds = Series(forest.predict(instances_enc), index = labels.index)
    sample_labels = Series(forest.predict(ds_container.X_train_enc), index = ds_container.y_train.index) # for train estimates

    print('Running Anchors on each instance and collecting results')
    # iterate through each instance to generate the anchors explanation
    results = [[]] * len(labels)
    for i in range(len(labels)):
        instance_id = labels.index[i]
        if i % 10 == 0: print('Working on ' + identifier + ' for instance ' + str(instance_id))

        # collect the results
        tc = preds.loc[instance_id]
        tc_lab = meta_data['get_label'](meta_data['class_col'], tc)

        # get test sample by leave-one-out on current instance
        _, _, loo_instances_enc, _, loo_true_labels = ds_container.get_loo_instances(instance_id, which_split='test')

        # get the model predicted labels
        loo_preds = Series(forest.predict(loo_instances_enc), index = loo_true_labels.index)

        # get the detail of the current index
        _, current_instance, current_instance_enc, _, current_instance_label = ds_container.get_by_id([instance_id], which_split='test') # matrix version of current instance

        explanation = Anchors_explanation(current_instance, anchors_explainer, forest,
                                            threshold=precis_threshold,
                                            random_state=random_state)

        current_instance = current_instance[0] # get it out of nested list (Anchors_explanation wanted it nested - nothing I can do about that)

        # Get train and test idx (boolean) covered by the anchor
        anchor_train_idx = np.array(np.all(ds_container.X_train_enc[:, explanation.features()] == current_instance[explanation.features()], axis=1).reshape(1, -1))[0]
        anchor_test_idx = np.array(np.all(loo_instances_enc[:, explanation.features()] == current_instance[explanation.features()], axis=1).reshape(1, -1))[0]

        # create a class to run the standard evaluation
        train_metrics = strcts.evaluator().evaluate(prior_labels=sample_labels, post_idx=anchor_train_idx)
        test_metrics = strcts.evaluator().evaluate(prior_labels=loo_preds, post_idx=anchor_test_idx)

        results[i] = [dataset_name,
            instance_id,
            identifier,
            ' AND '.join(explanation.names()),
            len(explanation.names()),
            tc,
            tc_lab,
            tc,
            tc_lab,
            np.array([tree.predict(current_instance_enc) for tree in forest.estimators_][0] == ci_forest_pred.values.mean())[0], # majority vote share
            test_metrics['prior']['p_counts'][tc],
            train_metrics['posterior'][tc],
            train_metrics['stability'][tc],
            train_metrics['recall'][tc],
            train_metrics['f1'][tc],
            train_metrics['accuracy'][tc],
            train_metrics['lift'][tc],
            train_metrics['coverage'],
            train_metrics['xcoverage'],
            test_metrics['posterior'][tc],
            test_metrics['stability'][tc],
            test_metrics['recall'][tc],
            test_metrics['f1'][tc],
            test_metrics['accuracy'][tc],
            test_metrics['lift'][tc],
            test_metrics['coverage'],
            test_metrics['xcoverage']]

    if save_path is not None:
        rt.save_results(results, save_results_path=save_path,
                        save_results_file=identifier + '_results' + '_rnst_' + str(random_state))

def defragTrees_prep(forest, meta_data,
                        X_train, y_train, X_test, y_test,
                        Kmax=10, maxitr=100, restart=10,
                        identifier='defragTrees', save_path=''):

    feature_names = meta_data['features_enc']
    class_names = meta_data['class_names_label_order']
    random_state = meta_data['random_state']

    print('Running defragTrees')
    print()
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

    rt.evaluate_model(y_true=y_test, y_pred=mdl.predict(np.array(X_test)),
                        class_names=class_names,
                        plot_cm=False, plot_cm_norm=False, # False here will output the metrics and suppress the plots
                        save_path=save_path,
                        identifier=identifier,
                        random_state=random_state)

    return(mdl)

def rule_list_from_dfrgtrs(dfrgtrs):
    rule_list = [[]] * len(dfrgtrs.rule_)
    for r, rule in enumerate(dfrgtrs.rule_):
        rule_list[r] = [(dfrgtrs.featurename_[item[0]-1], not item[1], item[2]) for item in rule]

    return(rule_list)

def which_rule(rule_list, X, features):
    left_idx = np.array([i for i in range(len(X.todense()))])
    rules_idx = np.full(len(left_idx), np.nan)
    rule_cascade = 0
    while len(left_idx) > 0:
        rule_evaluator = strcts.rule_evaluator()
        match_idx = rule_evaluator.apply_rule(rule=rule_list[rule_cascade],
                                              instances=X[left_idx,:],
                                              features=features)
        rules_idx[left_idx[match_idx]] = rule_cascade
        left_idx = np.array([li for li, mi in zip(left_idx, match_idx) if not mi])
        rule_cascade += 1
    rules_idx[np.where(np.isnan(rules_idx))] = rule_cascade + 1 # default prediction
    rules_idx = rules_idx.astype(int)
    return(rules_idx)

def defragTrees_benchmark(forest, ds_container, meta_data, dfrgtrs,
                            batch_size=100, n_instances=100,
                            save_path='', dataset_name='',
                            random_state=123):

    identifier = 'defragTrees'
    print('defragTrees benchmark')
    # OPTION 1 - batching (to be implemented in the new code, right now it will do just one batch)
    _, _, _, _, labels = unseen_data_prep(ds_container,
                                            n_instances=n_instances,
                                            batch_size=batch_size)

    coverage = np.zeros(len(labels))
    xcoverage = np.zeros(len(labels))
    learner_precision = np.zeros(len(labels))
    learner_stability = np.zeros(len(labels))
    learner_counts = np.zeros(len(labels))
    learner_recall = np.zeros(len(labels))
    learner_f1 = np.zeros(len(labels))
    learner_accu = np.zeros(len(labels))
    learner_lift = np.zeros(len(labels))
    forest_precision = np.zeros(len(labels))
    forest_stability = np.zeros(len(labels))
    forest_counts = np.zeros(len(labels))
    forest_recall = np.zeros(len(labels))
    forest_f1 = np.zeros(len(labels))
    forest_accu = np.zeros(len(labels))
    forest_lift = np.zeros(len(labels))

    rule_list = rule_list_from_dfrgtrs(dfrgtrs)

    results = [[]] * len(labels)
    evaluator = strcts.evaluator()
    for i in range(len(labels)):
        instance_id = labels.index[i]
        if i % 10 == 0: print('Working on ' + identifier + ' for instance ' + str(instance_id))

        # get test sample by leave-one-out on current instance
        _, _, loo_instances_enc, loo_instances_enc_matrix, loo_true_labels = ds_container.get_loo_instances(instance_id,
                                                                                                            which_split='test')

        # get the detail of the current index
        _, current_instance, current_instance_enc, current_instance_enc_matrix, current_instance_label = ds_container.get_by_id([instance_id]
                                                                                                      , which_split='test')

        # get predictions from forest and dfrgtrs
        forest_preds = Series(forest.predict(loo_instances_enc), index = loo_true_labels.index)
        dfrgtrs_preds = Series(dfrgtrs.predict(np.array(loo_instances_enc_matrix)), index = loo_true_labels.index)
        ci_forest_pred = Series(forest.predict(current_instance_enc), index = current_instance_label.index)
        ci_dfrgtrs_pred = Series(forest.predict(current_instance_enc_matrix), index = current_instance_label.index)

        # which rule appies to each loo instance
        rule_idx = which_rule(rule_list, loo_instances_enc, features=meta_data['features_enc'])

        # which rule appies to current instance
        rule = which_rule(rule_list, current_instance_enc, features=meta_data['features_enc'])

        # which instances are covered by this rule
        covered_instances = rule_idx == rule

        metrics = evaluator.evaluate(prior_labels=loo_true_labels,
                                        post_idx=covered_instances)

        # majority class is the forest vote class
        # target class is the benchmark algorithm prediction
        mc = ci_forest_pred.values
        tc = ci_dfrgtrs_pred.values

        results[i] = [dataset_name,
        instance_id,
        identifier,
        evaluator.prettify_rule(rule_list[rule[0]], meta_data['var_dict']),
        len(rule),
        mc,
        meta_data['get_label'](meta_data['class_col'], ci_forest_pred.values)[0],
        tc,
        meta_data['get_label'](meta_data['class_col'], ci_dfrgtrs_pred.values)[0],
        np.array([tree.predict(current_instance_enc) for tree in forest.estimators_][0] == ci_forest_pred.values.mean())[0],
        metrics['prior']['p_counts'][mc],
        metrics['posterior'][tc],
        metrics['stability'][tc],
        metrics['recall'][tc],
        metrics['f1'][tc],
        metrics['accuracy'][tc],
        metrics['lift'][tc],
        metrics['coverage'],
        metrics['xcoverage'],
        metrics['posterior'][mc],
        metrics['stability'][mc],
        metrics['recall'][mc],
        metrics['f1'][mc],
        metrics['accuracy'][mc],
        metrics['lift'][mc],
        metrics['coverage'],
        metrics['xcoverage']]

    if save_path is not None:
        rt.save_results(results, save_results_path=save_path,
                        save_results_file=identifier + '_results' + '_rnst_' + str(random_state))
