import json
import time
import timeit
import pickle
import numpy as np
import multiprocessing as mp
from pandas import DataFrame, Series
from forest_surveyor import p_count, p_count_corrected
import forest_surveyor.datasets as ds
from forest_surveyor.structures import forest_walker, batch_getter, rule_tester, loo_encoder
from forest_surveyor.routines import tune_rf, train_rf, evaluate_model, run_batches, anchors_preproc, anchors_explanation
from scipy.stats import chi2_contingency
from math import sqrt
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_recall_fscore_support, accuracy_score

# bug in sk-learn. Should be fixed in August
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
