import json
import time
import timeit
from pandas import Series
import CHIRPS.datasets as ds
import CHIRPS.routines as rt
import CHIRPS.structures as strcts

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
    # this will normalise the above parameters to the size of the dataset
    n_instances, n_batches = rt.batch_instance_ceiling(ds_container=ds_container, n_instances=n_instances, batch_size=batch_size)

    # this gets the next batch out of the data_split_container according to the required number of instances
    # all formats can be extracted, depending on the requirement
    # unencoded, encoded (sparse matrix is the type returned by scikit), ordinary dense matrix also available
    instances, instances_enc, instances_enc_matrix, labels = ds_container.get_next(batch_size, which_split='test') # default

    # OPTION 2 - just run with whole test set
    # instances = tt.X_test; instances_enc = tt.X_test_enc; instances_enc_matrix = tt.X_test_enc_matrix; labels = tt.y_test

    # Make all the model predictions from the decision forest
    preds = forest.predict(X=instances_enc)
    print()

    # 3.1 - Extract Tree Prediction Paths

    print('Walking forest for ' + str(len(labels)) + ' instances... (please wait)')

    # wrapper object needs the decision forest itself and the dataset meta data (we have a convenience function for this)
    f_walker = strcts.forest_walker(forest = forest, meta_data=meta_data)

    # set the timer
    forest_walk_start_time = timeit.default_timer()

    # do the walk - returns a batch_paths_container (even for just one instance)
    # requires the X instances in a matrix (dense, ordinary numpy matrix) - this is available in the data_split_container
    bp_container = f_walker.forest_walk(instances = instances_enc_matrix
                            , labels = preds # we're explaining the prediction, not the true label!
                            , forest_walk_async = forest_walk_async)

    # stop the timer
    forest_walk_end_time = timeit.default_timer()
    forest_walk_elapsed_time = forest_walk_end_time - forest_walk_start_time

    print('Forest Walk with async = ' + str(forest_walk_async))
    print('Forest Walk time elapsed:', "{:0.4f}".format(forest_walk_elapsed_time), 'seconds')
    print()

    # 3.2-3.4 - Freqent pattern mining of paths, Score and sort mined path segments, Merge path segments into one rule

    # build CHIRPS and a rule for each instance represented in the batch paths container
    CHIRPS = strcts.batch_CHIRPS_explainer(bp_container,
                                    forest=forest,
                                    sample_instances=ds_container.X_train_enc, # any representative sample can be used
                                    sample_labels=ds_container.y_train,  # any representative sample can be used
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

    # COMPARISON WITH OTHER METHODS
    # 1. Anchors

    print()
    print()
