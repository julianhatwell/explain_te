import CHIRPS.datasets as ds
import CHIRPS.routines as rt
import CHIRPS.structures as strcts
from CHIRPS import p_count_corrected
from CHIRPS import config as cfg

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_recall_fscore_support, accuracy_score

from CHIRPS import if_nexists_make_dir, if_nexists_make_file, chisq_indep_test, p_count_corrected
from CHIRPS.plotting import plot_confusion_matrix

from CHIRPS import config as cfg

import matplotlib.pyplot as plt
import pygraphviz as pgv
import networkx as nx
import pygraphviz
import matplotlib.image as img
import matplotlib.pyplot as plt

from sklearn.tree import export_graphviz
from StringIO import StringIO
from io import BytesIO

# bug in sk-learn. Should be fixed in August
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

def get_graph(dtc, n_classes, feat_names=None, size=[7, 7]):
    dot_file = StringIO()
    image_file = BytesIO()

    # Get the dot graph of our decision tree
    export_graphviz(dtc, out_file=dot_file, feature_names=feat_names, rounded=True, filled=True,
                    special_characters=True, class_names=map(str, range(n_classes)), max_depth=10)
    dot_file.seek(0)

    # Convert this dot graph into an image
    g = pygraphviz.AGraph(dot_file.read())
    g.layout('dot')
    # g.draw doesn't work when the image object doesn't have a name (with a proper extension)
    image_file.name = "image.png"
    image_file.seek(0)
    g.draw(path=image_file)
    image_file.seek(0)

    # Plot it
    plt.figure().set_size_inches(*size)
    plt.axis('off')
    plt.imshow(img.imread(fname=image_file))
    plt.show()
