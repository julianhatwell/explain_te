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

def forest_prep(X, y, save_path=None, class_names=None, override_tuning=False, tuning_grid=None, random_state=123, identifier='main'):

    if class_names is None:
        class_names=y.unique()

    best_params, _ = rt.tune_rf(
     X=X,
     y=y,
     grid=tuning_grid,
     save_path=save_path,
     override_tuning=override_tuning,
     random_state=random_state)

    rf = rt.RandomForestClassifier(random_state=random_state, oob_score=True, **best_params)
    rf.fit(X, y)

    # the outputs of this function are:
    # cm - confusion matrix as 2d array
    # acc - accuracy of model = correctly classified instance / total number instances
    # coka - Cohen's kappa score. Accuracy adjusted for probability of correct by random guess. Useful for multiclass problems
    # prfs - precision, recall, f-score, support with the following signatures as a 2d array
    # 0 <= p, r, f <= 1. s = number of instances for each true class label (row sums of cm)
    rt.evaluate_model(prediction_model=rf, X=X, y=y,
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
                    save_path='',
                    random_state=123):
    # 2. Prepare Unseen Data and Predictions
    print('Prepare Unseen Data and Predictions')
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
                                  save_results_file='results' + '_rnst_' + str(random_state),
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
                    precis_threshold=0.95,
                    random_state=123):
    # 2. Prepare Unseen Data and Predictions
    print('Prepare Unseen Data and Predictions')
    # OPTION 1 - batching (to be implemented in the new code, right now it will do just one batch)
    _, instances, instances_enc, instances_enc_matrix, labels = unseen_data_prep(ds_container,
                                                                                n_instances=n_instances,
                                                                                batch_size=batch_size)
    # get predictions
    preds = Series(forest.predict(instances_enc), index = labels.index)
    sample_labels = Series(forest.predict(ds_container.X_train_enc), index = ds_container.y_train.index) # for train estimates

    print('Running Anchors on each instance and collecting results')
    # iterate through each instance to generate the anchors explanation
    output_anch = [[]] * len(labels)
    for i in range(len(labels)):
        instance_id = labels.index[i]
        if i % 10 == 0: print('Working on Anchors for instance ' + str(instance_id))

        # get test sample by leave-one-out on current instance
        _, instances, instances_enc, _, labels = ds_container.get_loo_instances(instance_id) # matrix version of instances
        # get the model predicted labels
        labels = Series(forest.predict(instances_enc), index = labels.index)

        # get the detail of the current index
        _, current_instance, current_instance_enc, _, current_instance_label = ds_container.get_by_id([instance_id], which_split='test') # matrix version of current instance

        explanation = Anchors_explanation(current_instance, anchors_explainer, forest,
                                            threshold=precis_threshold,
                                            random_state=random_state)

        current_instance = current_instance[0] # get it out of nested list (which happens inside the Anchors_explanation - nothing I can do about that)

        # Get train and test examples where the anchor applies
        _, _, loo_instances_enc, _, loo_true_labels = ds_container.get_loo_instances(instance_id)

        anchor_train_idx = np.where(np.all(ds_container.X_train_enc[:, explanation.features()] == current_instance[explanation.features()], axis=1))[0]
        anchor_test_idx = np.where(np.all(loo_instances_enc[:, explanation.features()] == current_instance[explanation.features()], axis=1))[0]
        # fit_anchor_test_exclusive = [fat for fat in fit_anchor_test if fat != i] # exclude current instance

        # get the model predicted labels
        loo_preds = Series(forest.predict(loo_instances_enc), index = loo_true_labels.index)

        # collect the results
        tc = current_instance_label.values[0]
        tc_lab = meta_data['get_label'](meta_data['class_col'], tc)

        # train
        priors = p_count_corrected(sample_labels, [i for i in range(len(meta_data['class_names']))])
        if len(anchor_train_idx) > 0:
            p_counts = p_count_corrected(forest.predict(ds_container.X_train_enc[anchor_train_idx]), [i for i in range(len(meta_data['class_names']))])
        else:
            p_counts = p_count_corrected([None], [i for i in range(len(meta_data['class_names']))])

#                 counts = p_counts['counts']
#                 labels = p_counts['labels']
#                 post = p_counts['p_counts']
#                 p_corrected = np.array([p if p > 0.0 else 1.0 for p in post])
#                 cover = counts.sum() / priors['counts'].sum()
#                 recall = counts/priors['counts'] # recall
#                 r_corrected = np.array([r if r > 0.0 else 1.0 for r in recall]) # to avoid div by zeros
#                 observed = np.array((counts, priors['counts']))
#                 if counts.sum() > 0: # previous_counts.sum() == 0 is impossible
#                     chisq = chi2_contingency(observed=observed[:, np.where(observed.sum(axis=0) != 0)], correction=True)
#                 else:
#                     chisq = np.nan
#                 f1 = [2] * ((post * recall) / (p_corrected + r_corrected))
#                 not_covered_counts = counts + (np.sum(priors['counts']) - priors['counts']) - (np.sum(counts) - counts)
#                 accu = not_covered_counts/priors['counts'].sum()
#                 # to avoid div by zeros
#                 pri_corrected = np.array([pri if pri > 0.0 else 1.0 for pri in priors['p_counts']])
#                 pos_corrected = np.array([pos if pri > 0.0 else 0.0 for pri, pos in zip(priors['p_counts'], post)])
#                 if counts.sum() == 0:
#                     rec_corrected = np.array([0.0] * len(pos_corrected))
#                     cov_corrected = np.array([1.0] * len(pos_corrected))
#                 else:
#                     rec_corrected = counts / counts.sum()
#                     cov_corrected = np.array([counts.sum() / priors['counts'].sum()])

#                 lift = pos_corrected / ( ( cov_corrected ) * pri_corrected )

#                 # capture train
#                 mc = enc_rf.predict(tt['X_test'][i].reshape(1, -1))[0]
#                 mc_lab = mydata.class_names[enc_rf.predict(tt['X_test'][i].reshape(1, -1))[0]]
#                 tc = enc_rf.predict(tt['X_test'][i].reshape(1, -1))[0]
#                 tc_lab = mydata.class_names[enc_rf.predict(tt['X_test'][i].reshape(1, -1))[0]]
#                 vt = np.nan
#                 mvs = np.nan
#                 prior = priors['p_counts'][tc]
#                 prettify_rule = ' AND '.join(exp.names())
#                 rule_len = len(exp.names())
#                 tr_prec = post[tc]
#                 tr_recall = recall[tc]
#                 tr_f1 = f1[tc]
#                 tr_acc = accu[tc]
#                 tr_lift = lift[tc]
#                 tr_coverage = cover



    return()
