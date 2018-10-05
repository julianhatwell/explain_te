import numpy as np
from pathlib import Path as pth
from os import makedirs as mkdir

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

# create a directory if doesn't exist
def if_nexists_make_dir(save_path):
    if not pth(save_path).is_dir():
        mkdir(save_path)
