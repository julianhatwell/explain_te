from pandas import Series
import CHIRPS.datasets as ds

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

def export_data_splits(datasets, project_dir=None, random_state=123):
    # general data preparation
    for dataset in datasets:
        mydata = dataset(random_state=random_state, project_dir=project_dir)

        # train test split - one off hard-coded random state.
        # later we will vary the state to generate different forests for e.g. benchmarking
        train_index, test_index = mydata.get_tt_split_idx(random_state=123)
        tt = mydata.tt_split_by_idx(train_index, test_index).to_dict()
        # save to csv to use the same splits in methods that aren't available in Python
        mydata.tt_split_by_idx(train_index, test_index).to_csv(mydata.get_save_path(),
                                                                encoded_features = mydata.onehot_features)
