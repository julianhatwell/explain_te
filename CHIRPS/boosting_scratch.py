import math
import numpy as np
import CHIRPS.datasets as ds
import CHIRPS.routines as rt
import CHIRPS.structures as strcts
from CHIRPS import p_count_corrected
from CHIRPS import config as cfg

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
# from xgboost.sklearn import XGBClassifier # not compatible with latest scikit-learn versions, older versions break CHIRPS code at input matrix types
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_recall_fscore_support, accuracy_score

from CHIRPS import if_nexists_make_dir, if_nexists_make_file, chisq_indep_test, p_count_corrected
from CHIRPS.plotting import plot_confusion_matrix

from CHIRPS import config as cfg

# bug in sk-learn. Should be fixed in August
# import warnings
# warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
def calculate_weights(arr):
    def weight(err):
        err_value = (1-err)/err
        return(0.5 * math.log(err_value))

    vweight = np.vectorize(weight)

    return(vweight(arr))
