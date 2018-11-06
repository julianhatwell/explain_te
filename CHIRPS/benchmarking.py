import numpy as np
from copy import deepcopy
import CHIRPS.routines as rt
from lime import lime_tabular as limtab
from anchor import anchor_tabular as anchtab

def anchors_preproc(ds_container, meta_data):

    var_dict_anch = deepcopy(meta_data['var_dict'])
    ds_container_anch = deepcopy(ds_container)

    # create the discretise function from LIME tabular
    disc = limtab.QuartileDiscretizer(np.array(ds_container.X_train),
                                      categorical_features=meta_data['categorical_features'],
                                      feature_names=meta_data['features'])

    # create a copy of the var_dict with updated labels
    for vk in var_dict_anch.keys():
        if var_dict_anch[vk]['data_type'] == 'continuous':
            var_dict_anch[vk]['labels'] = disc.names[var_dict_anch[vk]['order_col']]
            var_dict_anch[vk]['data_type'] = 'discretised'

    var_dict_anch['categorical_names'] = {var_dict_anch[vk]['order_col'] : var_dict_anch[vk]['labels'] \
                                        for vk in var_dict_anch.keys() if not var_dict_anch[vk]['class_col']}

    # create discretised versions of the training and test data
    ds_container_anch.X_train = np.array(disc.discretize(np.array(ds_container.X_train)))
    ds_container_anch.X_test = np.array(disc.discretize(np.array(ds_container.X_test)))

        # fit the Anchors explainer. Onehot encode the data. Replace the data in the ds_container
    explainer = anchtab.AnchorTabularExplainer(meta_data['class_names'], meta_data['features'], ds_container_anch.X_train, var_dict_anch['categorical_names'])
    explainer.fit(ds_container_anch.X_train, ds_container_anch.y_train, ds_container_anch.X_test, ds_container_anch.y_test)

    ds_container_anch.X_train_enc = explainer.encoder.transform(ds_container_anch.X_train)
    ds_container_anch.X_test_enc = explainer.encoder.transform(ds_container_anch.X_test)

    ds_container_anch.X_train_enc_matrix = ds_container_anch.X_train_enc.todense()
    ds_container_anch.X_test_enc_matrix = ds_container_anch.X_test_enc.todense()

    return(ds_container_anch, var_dict_anch, explainer)

def anchors_explanation(instance, explainer, forest, random_state=123, threshold=0.95):
    np.random.seed(random_state)
    exp = explainer.explain_instance(instance, forest.predict, threshold=threshold)
    return(exp)

# # run anchors if requested
# anch_elapsed_time = None # optional no timing
# if run_anchors:
#     print('running anchors for random_state ' + str(mydata.random_state))
#     # collect timings
#     anch_start_time = timeit.default_timer()
#     instance_ids = tt['X_test'].index.tolist() # record of row indices will be lost after preproc
#     mydata, tt, explanation = anchors_preproc(dataset, random_state, iv_low, iv_high)
#
#     rf, enc_rf = train_rf(tt['X_train_enc'], y=tt['y_train'],
#     best_params=best_params,
#     encoder=tt['encoder'],
#     random_state=mydata.random_state)
#
#     # collect model prediction performance stats
#     if eval_model:
#         cm, acc, coka, prfs = evaluate_model(prediction_model=enc_rf, X=tt['X_test'], y=tt['y_test'],
#                      class_names=mydata.class_names,
#                      plot_cm=True, plot_cm_norm=True)
#     else:
#         cm, acc, coka, prfs = evaluate_model(prediction_model=enc_rf, X=tt['X_test'], y=tt['y_test'],
#                      class_names=mydata.class_names,
#                      plot_cm=False, plot_cm_norm=False)
#
#     # iterate through each instance to generate the anchors explanation
#     output_anch = [[]] * n_instances
#     for i in range(n_instances):
#         instance_id = instance_ids[i]
#         if i % 10 == 0: print('Working on Anchors for instance ' + str(instance_id))
#         instance = tt['X_test'][i]
#         exp = anchors_explanation(instance, explanation, rf, threshold=precis_threshold)
#         # capture the explainer
#         explainers[i].append(exp)
#
#         # Get test examples where the anchor applies
#         fit_anchor_train = np.where(np.all(tt['X_train'][:, exp.features()] == instance[exp.features()], axis=1))[0]
#         fit_anchor_test = np.where(np.all(tt['X_test'][:, exp.features()] == instance[exp.features()], axis=1))[0]
#         fit_anchor_test = [fat for fat in fit_anchor_test if fat != i] # exclude current instance
#
#         # train
#         priors = p_count_corrected(tt['y_train'], [i for i in range(len(mydata.class_names))])
#         if any(fit_anchor_train):
#             p_counts = p_count_corrected(enc_rf.predict(tt['X_train'][fit_anchor_train]), [i for i in range(len(mydata.class_names))])
#         else:
#             p_counts = p_count_corrected([None], [i for i in range(len(mydata.class_names))])
#         counts = p_counts['counts']
#         labels = p_counts['labels']
#         post = p_counts['p_counts']
#         p_corrected = np.array([p if p > 0.0 else 1.0 for p in post])
#         cover = counts.sum() / priors['counts'].sum()
#         recall = counts/priors['counts'] # recall
#         r_corrected = np.array([r if r > 0.0 else 1.0 for r in recall]) # to avoid div by zeros
#         observed = np.array((counts, priors['counts']))
#         if counts.sum() > 0: # previous_counts.sum() == 0 is impossible
#             chisq = chi2_contingency(observed=observed[:, np.where(observed.sum(axis=0) != 0)], correction=True)
#         else:
#             chisq = np.nan
#         f1 = [2] * ((post * recall) / (p_corrected + r_corrected))
#         not_covered_counts = counts + (np.sum(priors['counts']) - priors['counts']) - (np.sum(counts) - counts)
#         accu = not_covered_counts/priors['counts'].sum()
#         # to avoid div by zeros
#         pri_corrected = np.array([pri if pri > 0.0 else 1.0 for pri in priors['p_counts']])
#         pos_corrected = np.array([pos if pri > 0.0 else 0.0 for pri, pos in zip(priors['p_counts'], post)])
#         if counts.sum() == 0:
#             rec_corrected = np.array([0.0] * len(pos_corrected))
#             cov_corrected = np.array([1.0] * len(pos_corrected))
#         else:
#             rec_corrected = counts / counts.sum()
#             cov_corrected = np.array([counts.sum() / priors['counts'].sum()])
#
#         lift = pos_corrected / ( ( cov_corrected ) * pri_corrected )
#
#         # capture train
#         mc = enc_rf.predict(tt['X_test'][i].reshape(1, -1))[0]
#         mc_lab = mydata.class_names[enc_rf.predict(tt['X_test'][i].reshape(1, -1))[0]]
#         tc = enc_rf.predict(tt['X_test'][i].reshape(1, -1))[0]
#         tc_lab = mydata.class_names[enc_rf.predict(tt['X_test'][i].reshape(1, -1))[0]]
#         vt = np.nan
#         mvs = np.nan
#         prior = priors['p_counts'][tc]
#         prettify_rule = ' AND '.join(exp.names())
#         rule_len = len(exp.names())
#         tr_prec = post[tc]
#         tr_recall = recall[tc]
#         tr_f1 = f1[tc]
#         tr_acc = accu[tc]
#         tr_lift = lift[tc]
#         tr_coverage = cover
#
#         # test
#         priors = p_count_corrected(tt['y_test'], [i for i in range(len(mydata.class_names))])
#         if any(fit_anchor_test):
#             p_counts = p_count_corrected(enc_rf.predict(tt['X_test'][fit_anchor_test]), [i for i in range(len(mydata.class_names))])
#         else:
#             p_counts = p_count_corrected([None], [i for i in range(len(mydata.class_names))])
#         counts = p_counts['counts']
#         labels = p_counts['labels']
#         post = p_counts['p_counts']
#         p_corrected = np.array([p if p > 0.0 else 1.0 for p in post])
#         cover = counts.sum() / priors['counts'].sum()
#         recall = counts/priors['counts'] # recall
#         r_corrected = np.array([r if r > 0.0 else 1.0 for r in recall]) # to avoid div by zeros
#         observed = np.array((counts, priors['counts']))
#         if counts.sum() > 0: # previous_counts.sum() == 0 is impossible
#             chisq = chi2_contingency(observed=observed[:, np.where(observed.sum(axis=0) != 0)], correction=True)
#         else:
#             chisq = np.nan
#         f1 = [2] * ((post * recall) / (p_corrected + r_corrected))
#         not_covered_counts = counts + (np.sum(priors['counts']) - priors['counts']) - (np.sum(counts) - counts)
#         # accuracy = (TP + TN) / num_instances formula: https://books.google.co.uk/books?id=ubzZDQAAQBAJ&pg=PR75&lpg=PR75&dq=rule+precision+and+coverage&source=bl&ots=Aa4Gj7fh5g&sig=6OsF3y4Kyk9KlN08OPQfkZCuZOc&hl=en&sa=X&ved=0ahUKEwjM06aW2brZAhWCIsAKHY5sA4kQ6AEIUjAE#v=onepage&q=rule%20precision%20and%20coverage&f=false
#         accu = not_covered_counts/priors['counts'].sum()
#         pri_corrected = np.array([pri if pri > 0.0 else 1.0 for pri in priors['p_counts']]) # to avoid div by zeros
#         pos_corrected = np.array([pos if pri > 0.0 else 0.0 for pri, pos in zip(priors['p_counts'], post)]) # to avoid div by zeros
#         if counts.sum() == 0:
#             rec_corrected = np.array([0.0] * len(pos_corrected))
#             cov_corrected = np.array([1.0] * len(pos_corrected))
#         else:
#             rec_corrected = counts / counts.sum()
#             cov_corrected = np.array([counts.sum() / priors['counts'].sum()])
#
#         lift = pos_corrected / ( ( cov_corrected ) * pri_corrected )
#
#         # capture test
#         tt_prec = post[tc]
#         tt_recall = recall[tc]
#         tt_f1 = f1[tc]
#         tt_acc = accu[tc]
#         tt_lift = lift[tc]
#         tt_coverage = cover
#
#         output_anch[i] = [instance_id,
#                             'anchors', # result_set
#                             prettify_rule,
#                             rule_len,
#                             mc,
#                             mc_lab,
#                             tc,
#                             tc_lab,
#                             mvs,
#                             prior,
#                             tr_prec,
#                             tr_recall,
#                             tr_f1,
#                             tr_acc,
#                             tr_lift,
#                             tr_coverage,
#                             tt_prec,
#                             tt_recall,
#                             tt_f1,
#                             tt_acc,
#                             tt_lift,
#                             tt_coverage,
#                             acc,
#                             coka]
#
#     output = np.concatenate((output, output_anch), axis=0)
#     anch_end_time = timeit.default_timer()
#     anch_elapsed_time = anch_end_time - anch_start_time
#
# # save the tabular results to a file
# output_df = DataFrame(output, columns=headers)
# output_df.to_csv(mydata.make_save_path(mydata.pickle_dir.replace('pickles', 'results') + '_rnst_' + str(mydata.random_state) + "_addt_" + str(add_trees) + '_timetest.csv'))
# # save the full rule_acc_lite objects
# if save_rule_accs:
#     explainers_store = open(mydata.make_save_path('explainers' + '_rnst_' + str(mydata.random_state) + "_addt_" + str(add_trees) + '.pickle'), "wb")
#     pickle.dump(explainers, explainers_store)
#     explainers_store.close()
#
# print('Completed experiment for ' + str(dataset) + ':')
# print('random_state ' + str(mydata.random_state) + ' and ' +str(add_trees) + ' additional trees')
# # pass the elapsed times up to the caller
# return(wb_elapsed_time + wbres_elapsed_time, anch_elapsed_time, grid_idx)
