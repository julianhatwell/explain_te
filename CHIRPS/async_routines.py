import json
import time
import timeit
import numpy as np
from CHIRPS import p_count, p_count_corrected
import CHIRPS.datasets as ds
from CHIRPS.structures import rule_accumulator, forest_walker, rule_tester

from scipy.stats import chi2_contingency
from math import sqrt

def mine_path_segments(walked, data_container,
                        support_paths=0.1, alpha_paths=0.5,
                        disc_path_bins=4, disc_path_eqcounts=False,
                        which_trees='majority'):

    # discretize any numeric features
    walked.discretize_paths(data_container.var_dict,
                            bins=disc_path_bins,
                            equal_counts=disc_path_eqcounts)
    # the patterns are found but not scored and sorted yet
    walked.mine_patterns(support=support_paths)
    return(walked)

def score_sort_path_segments(walked, data_container,
                                sample_instances, sample_labels, encoder,
                                alpha_paths=0.5, weighting='chisq'):
    # best at -1 < alpha < 1
    # the patterns will be weighted by chi**2 for independence test, p-values
    if weighting == 'chisq':
        weights = [] * len(walked.patterns)
        for wp in walked.patterns:
            rt = rule_tester(data_container=data_container,
                            rule=wp,
                            sample_instances=sample_instances)
            rt.sample_instances = encoder.transform(rt.sample_instances)
            idx = rt.apply_rule()
            covered = p_count_corrected(sample_labels[idx], [i for i in range(len(data_container.class_names))])['counts']
            not_covered = p_count_corrected(sample_labels[~idx], [i for i in range(len(data_container.class_names))])['counts']
            observed = np.array((covered, not_covered))

            # this is the chisq based weighting. can add other options
            if covered.sum() > 0 and not_covered.sum() > 0: # previous_counts.sum() == 0 is impossible
                weights.append(sqrt(chi2_contingency(observed=observed[:, np.where(observed.sum(axis=0) != 0)], correction=True)[0]))
            else:
                weights.append(max(weights))

        # now the patterns are scored and sorted. alpha > 0 favours longer patterns. 0 neutral. < 0 shorter.
        walked.sort_patterns(alpha=alpha_paths, weights=weights) # with chi2 and support sorting
    else:
        walked.sort_patterns(alpha=alpha_paths) # with only support/alpha sorting
    return(walked)

def get_rule(rule_acc, encoder, sample_instances, sample_labels, pred_model,
                        algorithm='greedy_prec', precis_threshold=0.95):

        # run the rule accumulator with greedy precis
        rule_acc.build_rule(encoder=encoder,
                    sample_instances=sample_instances,
                    sample_labels=sample_labels,
                    algorithm=algorithm,
                    prediction_model=pred_model,
                    precis_threshold=precis_threshold)
        rule_acc.prune_rule()
        ra_lite = rule_acc.lite_instance()

        # collect completed rule accumulator
        return(ra_lite)

def as_chirps(walked, data_container,
                        sample_instances, sample_labels, encoder, pred_model,
                        support_paths=0.1, alpha_paths=0.5,
                        disc_path_bins=4, disc_path_eqcounts=False,
                        which_trees='majority', weighting='chisq',
                        algorithm='greedy_prec', precis_threshold=0.95,
                        batch_idx=None):
    # these steps make up the CHIRPS process:
    # mine paths for freq patts
    walked = mine_path_segments(walked, data_container,
                            support_paths, alpha_paths,
                            disc_path_bins, disc_path_eqcounts,
                            which_trees)
    # score and sort
    walked = score_sort_path_segments(walked, data_container,
                                    sample_instances, sample_labels,
                                    encoder, alpha_paths, weighting)
    # greedily add terms to create rule
    ra = rule_accumulator(data_container=data_container, paths_container=walked)
    ra_lite = get_rule(ra, encoder, sample_instances, sample_labels, pred_model,
    algorithm, precis_threshold)

    return(batch_idx, ra_lite)
