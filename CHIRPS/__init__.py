import numpy as np
from pathlib import Path as pth
from os import makedirs as mkdir
from scipy.stats import chi2_contingency

# helper function determines if we're in a jup notebook
def in_ipynb():
    try:
        cfg = get_ipython().config
        if len(cfg.keys()) > 0:
            if list(cfg.keys())[0]  == 'IPKernelApp':
                return(True)
            else:
                return(False)
        else:
            return(False)
    except NameError:
        return(False)

# helper function for returning counts and proportions of unique values in an array
def p_count(arr):
    labels, counts = np.unique(arr, return_counts = True)
    return(
    {'labels' : labels,
    'counts' : counts,
    'p_counts' : counts / len(arr)})

# insert any zeros for unrepresented classes
def p_count_corrected(arr, classes):
    n_classes = len(classes)
    p_counts = p_count(arr)
    pc = np.zeros(n_classes)
    c = np.zeros(n_classes, dtype=np.int64)
    for i, cn in enumerate(classes):
        if cn in p_counts['labels']:
            pc[i] = p_counts['p_counts'][np.where(p_counts['labels'] == cn)][0]
            c[i] = p_counts['counts'][np.where(p_counts['labels'] == cn)][0]
    return({'labels' : classes,
    'counts' : c,
    'p_counts' : pc})

def chisq_indep_test(counts, prior_counts):
    if type(counts) == list:
        counts = np.array(counts)
    observed = np.array((counts, prior_counts))
    if counts.sum() > 0: # previous_counts.sum() == 0 is impossible
        chisq = chi2_contingency(observed=observed[:, np.where(observed.sum(axis=0) != 0)], correction=True)[0:3]
    else:
        r, c = observed.shape
        chisq = (np.nan, np.nan, (r - 1) * (c - 1))
    return(chisq)

# create a directory if doesn't exist
def if_nexists_make_dir(save_path):
    if not pth(save_path).is_dir():
        mkdir(save_path)
