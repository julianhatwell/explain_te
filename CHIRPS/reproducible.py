import json
from pandas import Series
import CHIRPS.datasets as ds
import CHIRPS.routines as rt

datasets = [
            ds.adult_small_samp_data,
            ds.bankmark_samp_data,
            ds.car_data,
            ds.cardio_data,
            ds.credit_data,
            ds.german_data,
            ds.lending_tiny_samp_data,
            ds.nursery_samp_data,
            ds.rcdv_samp_data
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

def data_forest_prep(d_constructor, project_dir='', override_tuning=False,
                    tuning_grid=None, random_state=123, random_state_splits=123):
    mydata = d_constructor(random_state=random_state, project_dir=project_dir)
    meta_data = mydata.get_meta()
    save_path = meta_data['get_save_path']()
    train_index, test_index = mydata.get_tt_split_idx(random_state=random_state_splits)
    tt = mydata.tt_split(train_index, test_index)

    best_params, _ = rt.tune_rf(
     X=tt.X_train_enc,
     y=tt.y_train,
     grid=tuning_grid,
     save_path = save_path,
     override_tuning=override_tuning,
     random_state=mydata.random_state)

    rf = rt.RandomForestClassifier(random_state=random_state, oob_score=True, **best_params)
    rf.fit(X=tt.X_train_enc, y=tt.y_train)

    # the outputs of this function are:
    # cm - confusion matrix as 2d array
    # acc - accuracy of model = correctly classified instance / total number instances
    # coka - Cohen's kappa score. Accuracy adjusted for probability of correct by random guess. Useful for multiclass problems
    # prfs - precision, recall, f-score, support with the following signatures as a 2d array
    # 0 <= p, r, f <= 1. s = number of instances for each true class label (row sums of cm)
    cm, acc, coka, prfs = rt.evaluate_model(prediction_model=rf, X=tt.X_test_enc, y=tt.y_test,
                 class_names=meta_data['get_label'](meta_data['class_col']
                                                    , [i for i in range(len(meta_data['class_names']))]),
                 plot_cm=False, plot_cm_norm=False) # False here will output the metrics and suppress the plots

    test_metrics = {'confmat' : cm.tolist(),
                                'test_accuracy' : acc,
                                'test_kappa' : coka,
                                'test_prec' : prfs[0].tolist(),
                                'test_recall' : prfs[1].tolist(),
                                'test_f1' : prfs[2].tolist(),
                                'test_prior' : (prfs[3] / prfs[3].sum()).tolist(),
                                'test_posterior' : (cm.sum(axis=0) / prfs[3].sum()).tolist() }

    rt.update_model_performance(save_path=save_path, random_state=random_state, test_metrics=test_metrics)

    return(rf, tt, meta_data)
