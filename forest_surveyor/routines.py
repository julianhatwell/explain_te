import json
import time
import timeit
import pickle
import numpy as np
import multiprocessing as mp
from pandas import DataFrame, Series
from forest_surveyor import p_count, p_count_corrected
import forest_surveyor.datasets as ds
from forest_surveyor.plotting import plot_confusion_matrix
from forest_surveyor.structures import rule_accumulator, forest_walker, batch_getter, rule_tester, loo_encoder
from forest_surveyor.async_routines import as_chirps

from scipy.stats import chi2_contingency
from math import sqrt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_recall_fscore_support, accuracy_score

from anchor import anchor_tabular as anchtab
from lime import lime_tabular as limtab

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
        tree_start_time = timeit.default_timer()
        rf.set_params(oob_score = True, random_state=random_state, **g)
        rf.fit(X, y)
        tree_end_time = timeit.default_timer()
        g['fitting_time'] = tree_end_time - tree_start_time
        g['score'] = rf.oob_score_
        params.append(g)

    elapsed = timeit.default_timer() - start_time

    params = DataFrame(params).sort_values(['score','n_estimators','max_depth','min_samples_leaf'],
                                        ascending=[False, True, True, False])

    best_grid = params.loc[params['score'].idxmax()]
    best_params = {k: int(v) if k not in ('score', 'fitting_time') else v for k, v in best_grid.items()}

    if save_path is not None:
        with open(save_path + 'best_params_rndst_' + str(random_state) + '.json', 'w') as outfile:
            json.dump(best_params, outfile)

    return(best_params)

def tune_rf(X, y, grid = None, random_state=123, save_path = None, override_tuning=False):

    # to do - test allowable structure of grid input
    if override_tuning:
        print('Over-riding previous tuning parameters. New grid tuning... (please wait)')
        tun_start_time = timeit.default_timer()
        best_params = do_tuning(X, y, grid=grid, random_state=random_state, save_path=save_path)
        tun_elapsed_time = timeit.default_timer() - tun_start_time
        print('Tuning time elapsed:', "{:0.4f}".format(tun_elapsed_time), 'seconds')
    else:
        try:
            with open(save_path + 'best_params_rndst_' + str(random_state) + '.json', 'r') as infile:
                print('using previous tuning parameters')
                best_params = json.load(infile)
            return(best_params)
        except:
            print('finding best tuning parameters')
            best_params = do_tuning(X, y, grid=grid, random_state=random_state, save_path=save_path)

    print("Best OOB Accuracy Estimate during tuning: " "{:0.4f}".format(best_params['score']))
    print("Best parameters:", best_params)

    return(best_params)

def train_rf(X, y, best_params = None, encoder = None, random_state = 123):

    # to do - test allowable structure of grid input
    if best_params is not None:
        # train a random forest model using given parameters
        best_params = {k: v for k, v in best_params.items() if k not in ('score', 'fitting_time')}
        rf = RandomForestClassifier(random_state=random_state, oob_score=True, **best_params)
    else:
        # train a random forest model using default parameters
        rf = RandomForestClassifier(random_state=random_state, oob_score=True)

    rf.fit(X, y)

    if encoder is not None:
        # create helper function enc_model(). A pipeline: feature encoding -> rf model
        enc_model = make_pipeline(encoder, rf)
        return(rf, enc_model)
    else:
        # if no encoder provided, return the basic model
        return(rf, rf)

def evaluate_model(prediction_model, X, y, class_names=None, plot_cm=True, plot_cm_norm=True):
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
                              title='Normalized confusion matrix')
    return(cm, acc, coka, prfs)

def forest_survey(f_walker, X, y):

    if f_walker.encoder is not None:
        X = f_walker.encoder.transform(X)

    f_walker.full_survey(X, y)
    return(f_walker.forest_stats(np.unique(y)))

def run_batch_explanations(f_walker, getter,
 data_container, encoder, sample_instances, sample_labels,
 batch_size = 1, n_batches = 1,
 support_paths=0.1, alpha_paths=0.5,
 disc_path_bins=4, disc_path_eqcounts=False,
 alpha_scores=0.5, which_trees='majority',
 precis_threshold=0.95, weighting='chisq', greedy='greedy',
 forest_walk_async=False, chirps_explanation_async=False):

    pred_model = f_walker.prediction_model
    # create a list to collect completed rule accumulators
    completed_rule_accs = [[]] * (batch_size * n_batches)

    for b in range(n_batches):
        print('Walking forest for batch ' + str(b) + ' of batch size ' + str(batch_size) + '... (please wait)')
        instances, labels = getter.get_next(batch_size)
        instance_ids = labels.index.tolist()
        # get all the tree paths instance by instance
        forest_walk_start_time = timeit.default_timer()

        batch_walked = f_walker.forest_walk(instances = instances
                                , labels = labels
                                , async = forest_walk_async)

        forest_walk_end_time = timeit.default_timer()
        forest_walk_elapsed_time = forest_walk_end_time - forest_walk_start_time

        print('Forest Walk time elapsed:', "{:0.4f}".format(forest_walk_elapsed_time), 'seconds')
        print('Forest Walk with async = ' + str(forest_walk_async))

        ce_start_time = timeit.default_timer()
        if chirps_explanation_async:
            print('Running CHIRPS on batch... (please wait)')
            chp_start_time = timeit.default_timer()
            async_out = []
            n_cores = mp.cpu_count()-1
            pool = mp.Pool(processes=n_cores)
            for batch_idx in range(batch_size):
                instance_id = instance_ids[batch_idx]

                # extract the current instance paths for freq patt mining, filter by majority trees only
                walked = batch_walked.get_instance_paths(batch_idx, which_trees=which_trees)
                walked.instance_id = instance_id

                # run the chirps process on each instance paths
                async_out.append(pool.apply_async(as_chirps,
                    (walked, data_container,
                    sample_instances, sample_labels,
                    encoder, pred_model,
                    support_paths, alpha_paths,
                    disc_path_bins, disc_path_eqcounts,
                    which_trees, weighting,
                    greedy, precis_threshold,
                    batch_idx)
                ))

            # block and collect the pool
            pool.close()
            pool.join()

            # get the async results and sort to ensure original batch index order and remove batch index
            ce = [async_out[j].get() for j in range(len(async_out))]
            ce.sort()
            for batch_idx in range(batch_size):
                completed_rule_accs[b * batch_size + batch_idx] = [ce[batch_idx][1]] # embed object in list

            ce_end_time = timeit.default_timer()
            ce_elapsed_time = ce_end_time - ce_start_time

        else:
            for batch_idx in range(batch_size):
                instance_id = instance_ids[batch_idx]
                # extract the current instance paths for freq patt mining, filter by majority trees only
                walked = batch_walked.get_instance_paths(batch_idx, which_trees=which_trees)
                walked.instance_id = instance_id

                # run the chirps process on each instance paths
                _, completed_rule_acc = \
                    as_chirps(walked, data_container,
                    sample_instances, sample_labels,
                    encoder, pred_model,
                    support_paths, alpha_paths,
                    disc_path_bins, disc_path_eqcounts,
                    which_trees, weighting,
                    greedy, precis_threshold,
                    batch_idx)

                # add the finished rule accumulator to the results
                completed_rule_accs[b * batch_size + batch_idx] = [completed_rule_acc]
            ce_end_time = timeit.default_timer()
            ce_elapsed_time = ce_end_time - ce_start_time

        print('CHIRPS batch time elapsed:', "{:0.4f}".format(ce_elapsed_time), 'seconds')
        print('CHIRPS batch with async = ' + str(chirps_explanation_async))


    algorithm = ['greedy_prec'] # this method proved to be the best. For alternatives, go to the github and see older versions
    return(completed_rule_accs, algorithm)

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
