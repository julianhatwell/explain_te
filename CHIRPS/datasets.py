import urllib
import numpy as np
import pandas as pd

from CHIRPS import config as cfg
from CHIRPS.structures import data_container

# accident
def accident(random_state=123, project_dir=None):

    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'accident.csv.gz',
                        compression='gzip'),
    class_col = 'Accident_Severity',
    project_dir = project_dir,
    save_dir = 'accident',
    random_state=random_state,
    spiel = '''
    Data Set Information:
    These files provide detailed road safety data about the circumstances of personal injury road accidents in GB from 1979, the types (including Make and Model) of vehicles involved and the consequential casualties. The statistics relate only to personal injury accidents on public roads that are reported to the police, and subsequently recorded, using the STATS19 accident reporting form.

    All the data variables are coded rather than containing textual strings. The lookup tables are available in the "Additional resources" section towards the bottom of the table.

    Please note that the 2015 data were revised on the 29th September 2016.

    Accident, Vehicle and Casualty data for 2005 - 2009 are available in the time series files under 2014. Data for 1979 - 2004 are available as a single download under 2004 below.

    Also includes: Results of breath-test screening data from recently introduced digital breath testing devices, as provided by Police Authorities in England and Wales

    Results of blood alcohol levels (milligrams / 100 millilitres of blood) provided by matching coroners’ data (provided by Coroners in England and Wales and by Procurators Fiscal in Scotland) with fatality data from the STATS19 police data of road accidents in Great Britain. For cases when the Blood Alcohol Levels for a fatality are "unknown" are a consequence of an unsuccessful match between the two data sets.

    Data clean up by James Brooke
    ''')
    return(data_cont)

# accident sample: 0.1
def accident_samp(random_state=123, project_dir=None):

    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'accident_samp.csv.gz',
                        compression='gzip'),
    class_col = 'Accident_Severity',
    project_dir = project_dir,
    save_dir = 'accident_samp',
    random_state=random_state,
    spiel = '''
    Data Set Information:
    These files provide detailed road safety data about the circumstances of personal injury road accidents in GB from 1979, the types (including Make and Model) of vehicles involved and the consequential casualties. The statistics relate only to personal injury accidents on public roads that are reported to the police, and subsequently recorded, using the STATS19 accident reporting form.

    All the data variables are coded rather than containing textual strings. The lookup tables are available in the "Additional resources" section towards the bottom of the table.

    Please note that the 2015 data were revised on the 29th September 2016.

    Accident, Vehicle and Casualty data for 2005 - 2009 are available in the time series files under 2014. Data for 1979 - 2004 are available as a single download under 2004 below.

    Also includes: Results of breath-test screening data from recently introduced digital breath testing devices, as provided by Police Authorities in England and Wales

    Results of blood alcohol levels (milligrams / 100 millilitres of blood) provided by matching coroners’ data (provided by Coroners in England and Wales and by Procurators Fiscal in Scotland) with fatality data from the STATS19 police data of road accidents in Great Britain. For cases when the Blood Alcohol Levels for a fatality are "unknown" are a consequence of an unsuccessful match between the two data sets.

    Data clean up by James Brooke
    ''')
    return(data_cont)

# accident small sample: 0.01
def accident_small_samp(random_state=123, project_dir=None):

    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'accident_small_samp.csv.gz',
                        compression='gzip'),
    class_col = 'Accident_Severity',
    project_dir = project_dir,
    save_dir = 'accident_small_samp',
    random_state=random_state,
    spiel = '''
    Data Set Information:
    These files provide detailed road safety data about the circumstances of personal injury road accidents in GB from 1979, the types (including Make and Model) of vehicles involved and the consequential casualties. The statistics relate only to personal injury accidents on public roads that are reported to the police, and subsequently recorded, using the STATS19 accident reporting form.

    All the data variables are coded rather than containing textual strings. The lookup tables are available in the "Additional resources" section towards the bottom of the table.

    Please note that the 2015 data were revised on the 29th September 2016.

    Accident, Vehicle and Casualty data for 2005 - 2009 are available in the time series files under 2014. Data for 1979 - 2004 are available as a single download under 2004 below.

    Also includes: Results of breath-test screening data from recently introduced digital breath testing devices, as provided by Police Authorities in England and Wales

    Results of blood alcohol levels (milligrams / 100 millilitres of blood) provided by matching coroners’ data (provided by Coroners in England and Wales and by Procurators Fiscal in Scotland) with fatality data from the STATS19 police data of road accidents in Great Britain. For cases when the Blood Alcohol Levels for a fatality are "unknown" are a consequence of an unsuccessful match between the two data sets.

    Data clean up by James Brooke
    ''')
    return(data_cont)

# adult
def adult(random_state=123, project_dir=None):
    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'adult.csv.gz',
                        compression='gzip'),
    class_col = 'income',
    project_dir = project_dir,
    save_dir = 'adult',
    random_state=random_state,
    spiel = '''
    Data Description:
    This data was extracted from the adult bureau database found at
    http://www.adult.gov/ftp/pub/DES/www/welcome.html
    Donor: Ronny Kohavi and Barry Becker,
          Data Mining and Visualization
          Silicon Graphics.
          e-mail: ronnyk@sgi.com for questions.
    Split into train-test using MLC++ GenCVFiles (2/3, 1/3 random).
    48842 instances, mix of continuous and discrete    (train=32561, test=16281)
    45222 if instances with unknown values are removed (train=30162, test=15060)
    Duplicate or conflicting instances : 6
    Class probabilities for adult.all file
    Probability for the label '>50K'  : 23.93% / 24.78% (without unknowns)
    Probability for the label '<=50K' : 76.07% / 75.22% (without unknowns)
    Extraction was done by Barry Becker from the 1994 adult database.  A set of
     reasonably clean records was extracted using the following conditions:
     ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    ''')
    return(data_cont)

# adult sample: 0.25
def adult_samp(random_state=123, project_dir=None):
    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'adult_samp.csv.gz',
                        compression='gzip'),
    class_col = 'income',
    project_dir = project_dir,
    save_dir = 'adult_samp',
    random_state=random_state,
    spiel = '''
    Data Description:
    This data was extracted from the adult bureau database found at
    http://www.adult.gov/ftp/pub/DES/www/welcome.html
    Donor: Ronny Kohavi and Barry Becker,
          Data Mining and Visualization
          Silicon Graphics.
          e-mail: ronnyk@sgi.com for questions.
    Split into train-test using MLC++ GenCVFiles (2/3, 1/3 random).
    48842 instances, mix of continuous and discrete    (train=32561, test=16281)
    45222 if instances with unknown values are removed (train=30162, test=15060)
    Duplicate or conflicting instances : 6
    Class probabilities for adult.all file
    Probability for the label '>50K'  : 23.93% / 24.78% (without unknowns)
    Probability for the label '<=50K' : 76.07% / 75.22% (without unknowns)
    Extraction was done by Barry Becker from the 1994 adult database.  A set of
     reasonably clean records was extracted using the following conditions:
     ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    ''')
    return(data_cont)

# adult sample: 0.025
def adult_small_samp(random_state=123, project_dir=None):
    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'adult_small_samp.csv.gz',
                        compression='gzip'),
    class_col = 'income',
    project_dir = project_dir,
    save_dir = 'adult_small_samp',
    random_state=random_state,
    spiel = '''
    Data Description:
    This data was extracted from the adult bureau database found at
    http://www.adult.gov/ftp/pub/DES/www/welcome.html
    Donor: Ronny Kohavi and Barry Becker,
          Data Mining and Visualization
          Silicon Graphics.
          e-mail: ronnyk@sgi.com for questions.
    Split into train-test using MLC++ GenCVFiles (2/3, 1/3 random).
    48842 instances, mix of continuous and discrete    (train=32561, test=16281)
    45222 if instances with unknown values are removed (train=30162, test=15060)
    Duplicate or conflicting instances : 6
    Class probabilities for adult.all file
    Probability for the label '>50K'  : 23.93% / 24.78% (without unknowns)
    Probability for the label '<=50K' : 76.07% / 75.22% (without unknowns)
    Extraction was done by Barry Becker from the 1994 adult database.  A set of
     reasonably clean records was extracted using the following conditions:
     ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    ''')
    return(data_cont)

# bank marketing
def bankmark(random_state=123, project_dir=None):

    vtypes = {'age': np.int16,
     'campaign': np.int16,
     'cons.conf.idx': np.float16,
     'cons.price.idx': np.float16,
     'contact': object,
     'day_of_week': object,
     'default': object,
     'duration': np.int16,
     'education': object,
     'emp.var.rate': np.float16,
     'euribor3m': np.float16,
     'housing': object,
     'job': object,
     'loan': object,
     'marital': object,
     'month': object,
     'nr.employed': np.float16,
     'pdays': np.int16,
     'poutcome': object,
     'previous': np.int16,
     'y': object}

    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'bankmark.csv.gz',
                        compression='gzip',
                        dtype=vtypes),
    class_col = 'y',
    project_dir = project_dir,
    save_dir = 'bankmark',
    random_state=random_state,
    spiel = '''
    Data Set Information:
    The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

    There are four datasets:
    1) bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
    2) bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.
    3) bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs).
    4) bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs).
    The smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM).

    The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).


    Attribute Information:

    Input variables:
    # bank client data:
    1 - age (numeric)
    2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
    3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
    4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
    5 - default: has credit in default? (categorical: 'no','yes','unknown')
    6 - housing: has housing loan? (categorical: 'no','yes','unknown')
    7 - loan: has personal loan? (categorical: 'no','yes','unknown')
    # related with the last contact of the current campaign:
    8 - contact: contact communication type (categorical: 'cellular','telephone')
    9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
    10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
    11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
    # other attributes:
    12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
    13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
    14 - previous: number of contacts performed before this campaign and for this client (numeric)
    15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
    # social and economic context attributes
    16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
    17 - cons.price.idx: consumer price index - monthly indicator (numeric)
    18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
    19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
    20 - nr.employed: number of employees - quarterly indicator (numeric)

    Output variable (desired target):
    21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
    ''')
    return(data_cont)

# bank marketing sample: 0.05
def bankmark_samp(random_state=123, project_dir=None):

    vtypes = {'age': np.int16,
     'campaign': np.int16,
     'cons.conf.idx': np.float16,
     'cons.price.idx': np.float16,
     'contact': object,
     'day_of_week': object,
     'default': object,
     'duration': np.int16,
     'education': object,
     'emp.var.rate': np.float16,
     'euribor3m': np.float16,
     'housing': object,
     'job': object,
     'loan': object,
     'marital': object,
     'month': object,
     'nr.employed': np.float16,
     'pdays': np.int16,
     'poutcome': object,
     'previous': np.int16,
     'y': object}

    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'bankmark_samp.csv.gz',
                        compression='gzip',
                        dtype=vtypes),
    class_col = 'y',
    project_dir = project_dir,
    save_dir = 'bankmark_samp',
    random_state=random_state,
    spiel = '''
    Data Set Information:
    The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

    There are four datasets:
    1) bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
    2) bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.
    3) bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs).
    4) bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs).
    The smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM).

    The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).


    Attribute Information:

    Input variables:
    # bank client data:
    1 - age (numeric)
    2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
    3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
    4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
    5 - default: has credit in default? (categorical: 'no','yes','unknown')
    6 - housing: has housing loan? (categorical: 'no','yes','unknown')
    7 - loan: has personal loan? (categorical: 'no','yes','unknown')
    # related with the last contact of the current campaign:
    8 - contact: contact communication type (categorical: 'cellular','telephone')
    9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
    10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
    11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
    # other attributes:
    12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
    13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
    14 - previous: number of contacts performed before this campaign and for this client (numeric)
    15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
    # social and economic context attributes
    16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
    17 - cons.price.idx: consumer price index - monthly indicator (numeric)
    18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
    19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
    20 - nr.employed: number of employees - quarterly indicator (numeric)

    Output variable (desired target):
    21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
    ''')
    return(data_cont)

# car
def car(random_state=123, project_dir=None):
    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'car.csv.gz',
                        compression='gzip'),
    class_col = 'acceptability',
    var_names = ['buying'
                , 'maint'
                , 'doors'
                , 'persons'
                , 'lug_boot'
                , 'safety'
                , 'acceptability'],
    project_dir = project_dir,
    save_dir = 'car',
    random_state=random_state,
    spiel = '''
    M. Bohanec and V. Rajkovic: Knowledge acquisition and explanation for
    multi-attribute decision making. In 8th Intl Workshop on Expert
    Systems and their Applications, Avignon, France. pages 59-78, 1988.

    Within machine-learning, this dataset was used for the evaluation
    of HINT (Hierarchy INduction Tool), which was proved to be able to
    completely reconstruct the original hierarchical model. This,
    together with a comparison with C4.5, is presented in

    B. Zupan, M. Bohanec, I. Bratko, J. Demsar: Machine learning by
    function decomposition. ICML-97, Nashville, TN. 1997 (to appear)
    ''')
    return(data_cont)

# cardio
def cardio(random_state=123, project_dir=None):
    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'cardio.csv.gz',
                        compression='gzip'),
    class_col = 'NSP',
    project_dir = project_dir,
    save_dir = 'cardio',
    random_state=random_state,
    spiel = '''
    Data Set Information:
    2126 fetal cardiotocograms (CTGs) were automatically processed and the respective diagnostic features measured. The CTGs were also classified by three expert obstetricians and a consensus classification label assigned to each of them. Classification was both with respect to a morphologic pattern (A, B, C. ...) and to a fetal state (N, S, P). Therefore the dataset can be used either for 10-class or 3-class experiments.


    Attribute Information:
    LB - FHR baseline (beats per minute)
    AC - # of accelerations per second
    FM - # of fetal movements per second
    UC - # of uterine contractions per second
    DL - # of light decelerations per second
    DS - # of severe decelerations per second
    DP - # of prolongued decelerations per second
    ASTV - percentage of time with abnormal short term variability
    MSTV - mean value of short term variability
    ALTV - percentage of time with abnormal long term variability
    MLTV - mean value of long term variability
    Width - width of FHR histogram
    Min - minimum of FHR histogram
    Max - Maximum of FHR histogram
    Nmax - # of histogram peaks
    Nzeros - # of histogram zeros
    Mode - histogram mode
    Mean - histogram mean
    Median - histogram median
    Variance - histogram variance
    Tendency - histogram tendency
    CLASS - FHR pattern class code (1 to 10) # alternative class
    NSP - fetal state class code (N=normal; S=suspect; P=pathologic)
    ''')
    return(data_cont)

# credit
def credit(random_state=123, project_dir=None):
    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'credit.csv.gz',
                    compression='gzip'),
    class_col = 'A16',
    project_dir = project_dir,
    save_dir = 'credit',
    random_state=random_state,
    spiel = '''
    Data Set Information:

    This file concerns credit card applications. All attribute names and values have been changed to meaningless symbols to protect confidentiality of the data.

    This dataset is interesting because there is a good mix of attributes -- continuous, nominal with small numbers of values, and nominal with larger numbers of values. There are also a few missing values.

    Attribute Information:

    A1:	b, a.
    A2:	continuous.
    A3:	continuous.
    A4:	u, y, l, t.
    A5:	g, p, gg.
    A6:	c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
    A7:	v, h, bb, j, n, z, dd, ff, o.
    A8:	continuous.
    A9:	t, f.
    A10:	t, f.
    A11:	continuous.
    A12:	t, f.
    A13:	g, p, s.
    A14:	continuous.
    A15:	continuous.
    A16: +,- (class attribute)
    ''')

    return(data_cont)

# german
def german(random_state=123, project_dir=None):
    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'german.csv.gz',
                    compression='gzip'),
    class_col = 'rating',
    save_dir = 'german',
    project_dir = project_dir,
    random_state=random_state,
    spiel = '''
    Source:
    Professor Dr. Hans Hofmann
    Institut f"ur Statistik und "Okonometrie
    Universit"at Hamburg
    FB Wirtschaftswissenschaften
    Von-Melle-Park 5
    2000 Hamburg 13

    Data Set Information:
    Two datasets are provided. the original dataset, in the form provided by Prof. Hofmann, contains categorical/symbolic attributes and is in the file "german.data".
    For algorithms that need numerical attributes, Strathclyde University produced the file "german.data-numeric". This file has been edited and several indicator variables added to make it suitable for algorithms which cannot cope with categorical variables. Several attributes that are ordered categorical (such as attribute 17) have been coded as integer. This was the form used by StatLog.

    This dataset requires use of a cost matrix:
    . 1 2
    ------
    1 0 1
    -----
    2 5 0

    (1 = Good, 2 = Bad)
    The rows represent the actual classification and the columns the predicted classification.
    It is worse to class a customer as good when they are bad (5), than it is to class a customer as bad when they are good (1).
    ''')

    return(data_cont)

# lending
def lending(random_state=123, project_dir=None):
    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'lending.csv.gz',
                    compression='gzip'),
    class_col = 'loan_status',
    project_dir = project_dir,
    save_dir = 'lending',
    random_state=random_state,
    spiel = '''
    Data Set Information:
    Orignates from: https://www.lendingclub.com/info/download-data.action

    See also:
    https://www.kaggle.com/wordsforthewise/lending-club

    Prepared by Nate George: https://github.com/nateGeorge/preprocess_lending_club_data
    ''')

    return(data_cont)

# lending sample: 0.1
def lending_samp(random_state=123, project_dir=None):
    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'lending_samp.csv.gz',
                    compression='gzip'),
    class_col = 'loan_status',
    project_dir = project_dir,
    save_dir = 'lending_samp',
    random_state=random_state,
    spiel = '''
    Data Set Information:
    Orignates from: https://www.lendingclub.com/info/download-data.action

    See also:
    https://www.kaggle.com/wordsforthewise/lending-club

    Prepared by Nate George: https://github.com/nateGeorge/preprocess_lending_club_data
    ''')

    return(data_cont)

# lending small sample: 0.01
def lending_small_samp(random_state=123, project_dir=None):
    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'lending_small_samp.csv.gz',
                    compression='gzip'),
    class_col = 'loan_status',
    project_dir = project_dir,
    save_dir = 'lending_small_samp',
    random_state=random_state,
    spiel = '''
    Data Set Information:
    Orignates from: https://www.lendingclub.com/info/download-data.action

    See also:
    https://www.kaggle.com/wordsforthewise/lending-club

    Prepared by Nate George: https://github.com/nateGeorge/preprocess_lending_club_data
    ''')

    return(data_cont)

# lending tiny sample: 0.0025
def lending_tiny_samp(random_state=123, project_dir=None):
    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'lending_tiny_samp.csv.gz',
                    compression='gzip'),
    class_col = 'loan_status',
    project_dir = project_dir,
    save_dir = 'lending_tiny_samp',
    random_state=random_state,
    spiel = '''
    Data Set Information:
    Orignates from: https://www.lendingclub.com/info/download-data.action

    See also:
    https://www.kaggle.com/wordsforthewise/lending-club

    Prepared by Nate George: https://github.com/nateGeorge/preprocess_lending_club_data
    ''')

    return(data_cont)

# nursery
def nursery(random_state=123, project_dir=None):
    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'nursery.csv.gz',
                    compression='gzip'),
    class_col = 'decision',
    project_dir = project_dir,
    save_dir = 'nursery',
    random_state=random_state,
    spiel = '''
    Data Description:
    Nursery Database was derived from a hierarchical decision model
    originally developed to rank applications for nursery schools. It
    was used during several years in 1980's when there was excessive
    enrollment to these schools in Ljubljana, Slovenia, and the
    rejected applications frequently needed an objective
    explanation. The final decision depended on three subproblems:
    occupation of parents and child's nursery, family structure and
    financial standing, and social and health picture of the family.
    The model was developed within expert system shell for decision
    making DEX (M. Bohanec, V. Rajkovic: Expert system for decision
    making. Sistemica 1(1), pp. 145-157, 1990.).
    ''')

    return(data_cont)

# nursery sample: 0.2
def nursery_samp(random_state=123, project_dir=None):
    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'nursery_samp.csv.gz',
                    compression='gzip'),
    class_col = 'decision',
    project_dir = project_dir,
    save_dir = 'nursery_samp',
    random_state=random_state,
    spiel = '''
    Data Description:
    Nursery Database was derived from a hierarchical decision model
    originally developed to rank applications for nursery schools. It
    was used during several years in 1980's when there was excessive
    enrollment to these schools in Ljubljana, Slovenia, and the
    rejected applications frequently needed an objective
    explanation. The final decision depended on three subproblems:
    occupation of parents and child's nursery, family structure and
    financial standing, and social and health picture of the family.
    The model was developed within expert system shell for decision
    making DEX (M. Bohanec, V. Rajkovic: Expert system for decision
    making. Sistemica 1(1), pp. 145-157, 1990.).
    ''')

    return(data_cont)

# rcdv
def rcdv(random_state=123, project_dir=None):
    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'rcdv.csv.gz',
                    compression='gzip'),
    class_col = 'recid',
    project_dir = project_dir,
    save_dir = 'rcdv',
    random_state=random_state,
    spiel = '''
    Data Set Information:
    This is a description of the data on the file, DATA1978.
    The description was prepared by Peter Schmidt, Department of Economics, Michigan State University, East Lansing, Michigan 48824.
    The data were gathered as part of a grant from the National Institute of Justice to Peter Schmidt and Ann Witte, “Improving Predictions of Recidivism by Use of Individual Characteristics,” 84-IJ-CX-0021.
    A more complete description of the data, and of the uses to which they were put, can be found in the final report for this grant.
    Another similar dataset, contained in a file DATA1980 on a separate diskette, is also described in that report.

    The North Carolina Department of Correction furnished a data tape which was to contain information on all individuals released from a North Carolina prison during the period from July 1, 1977 through June 30, 1978.
    There were 9457 individual records on this tape. However, 130 records were deleted because of obvious defects.
    In almost all cases, the reason for deletion is that the individual’s date of release was in fact not during the time period which defined the data set.
    This left a total of 9327 individual records, and accordingly there are 9327 records on DATA1978.

    The basic sample of 9327 observations contained many observations for which one or more of the variables used in our analyses were missing.
    Specifically, 4709 observations were missing information on one or more such variables, and these 4709 observations constitute the “missing data” file.
    The other 4618 observations which contained complete information were randomly split into an “analysis file” of 1540 observations and a “validation file” of 3078 observations.

    DATA 1978 contains 9327 individual records. Each individual record contains 28 columns of data, representing the following 19 variables.

    WHITE ALCHY JUNKY SUPER MARRIED FELON WORKREL PROPTY PERSON MALE PRIORS SCHOOL RULE AGE TSERVD FOLLOW RECID TIME FILE
    1 2 3 4 5 6 7 8 9 10 11-12 13-14 15-16 17-19 20-22 23-24 25-27 28

    WHITE is a dummy (indicator) variable equal to zero if the individual is black, and equal to one otherwise. Basically, WHITE equals one for whites and zero for blacks. However, the North Carolina prison population also contains a small number of Native Americans, Hispanics, Orientals, and individuals of “other” race. They are treated as whites, by the above definition.
    ALCHY is a dummy variable equal to one if the individual’s record indicates a serious problem with alcohol, and equal to zero otherwise. It is important to note that for individuals in the missing data sample (FILE = 3), the value of ALCHY is recorded as zero, but is meaningless.
    JUNKY is a dummy variable equal to one if the individual’s record indicates use of hard drugs, and equal to zero otherwise. It is important to note that for individuals in the missing data sample (FILE = 3), the value of JUNKY is recorded as zero, but is meaningless.
    SUPER is a dummy variable equal to one if the individual’s release from the sample sentence was supervised (e.g., parole), and equal to zero otherwise.
    MARRIED is a dummy variable equal to one if the individual was married at the time of release from the sample sentence, and equal to zero otherwise.
    FELON is a dummy variable equal to one if the sample conviction was for a felony, and equal to zero if it was for a misdemeanor.
    WORKREL is a dummy variable equal to one if the individual participated in the North Carolina prisoner work release program during the sample sentence, and equal to zero otherwise.
    PROPTY is a dummy variable equal to one if the sample conviction was for a crime against property, and equal to zero otherwise. A detailed listing of the crime codes which define this variable (and PERSON below) can be found in A. Witte, Work Release in North Carolina: An Evaluation of Its Post Release Effects, Chapel Hill, North Carolina: Institute for Research in Social Science.
    PERSON is a dummy variable equal to one if the sample conviction was for a crime against a person, and equal to zero otherwise. (Incidentally, note that PROPTY plus PERSON is not necessarily equal to one, because there is an additional miscellaneous category of offenses which are neither offenses against property nor offenses against a person.)
    MALE is a dummy variable equal to one if the individual is male, and equal to zero if the individual is female.
    PRIORS is the number of previous incarcerations, not including the sample sentence. The value -9 indicates that this information is missing.
    SCHOOL is the number of years of formal schooling completed. The value zero indicates that this information is missing.
    RULE is the number of prison rule violations reported during the sample sentence.
    AGE is age (in months) at time of release.
    TSERVD is the time served (in months) for the sample sentence.
    FOLLOW is the length of the followup period, in months. (The followup period is the time from relase until the North Carolina Department of Correction records were searched, in April, 1984.)
    RECID is a dummy variable equal to one if the individual returned to a North Carolina prison during the followup period, and equal to zero otherwise.
    TIME is the length of time from release from the sample sentence until return to prison in North Carolina, for individuals for whom RECID equals one. TIME is rounded to the nearest month. (In particular, note that TIME equals zero for individuals who return to prison in North Carolina within the first half month after release.) For individuals for whom RECID equals zero, the value of TIME is meaningless. For such individuals, TIME is usually recorded as zero, but it is occasionally recorded as the length of the followup period. We emphasize again that neither value is meaningful, for those individuals for whom RECID equals zero.
    FILE is a variable indicating to which data sample the individual record belongs. The value 1 indicates the analysis sample, 2 the validation sampel and 3 is missing data sample.
    ''')

    return(data_cont)

# rcdv sample: 0.1
def rcdv_samp(random_state=123, project_dir=None):
    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'rcdv_samp.csv.gz',
                    compression='gzip'),
    class_col = 'recid',
    project_dir = project_dir,
    save_dir = 'rcdv_samp',
    random_state=random_state,
    spiel = '''
    Data Set Information:
    This is a description of the data on the file, DATA1978.
    The description was prepared by Peter Schmidt, Department of Economics, Michigan State University, East Lansing, Michigan 48824.
    The data were gathered as part of a grant from the National Institute of Justice to Peter Schmidt and Ann Witte, “Improving Predictions of Recidivism by Use of Individual Characteristics,” 84-IJ-CX-0021.
    A more complete description of the data, and of the uses to which they were put, can be found in the final report for this grant.
    Another similar dataset, contained in a file DATA1980 on a separate diskette, is also described in that report.

    The North Carolina Department of Correction furnished a data tape which was to contain information on all individuals released from a North Carolina prison during the period from July 1, 1977 through June 30, 1978.
    There were 9457 individual records on this tape. However, 130 records were deleted because of obvious defects.
    In almost all cases, the reason for deletion is that the individual’s date of release was in fact not during the time period which defined the data set.
    This left a total of 9327 individual records, and accordingly there are 9327 records on DATA1978.

    The basic sample of 9327 observations contained many observations for which one or more of the variables used in our analyses were missing.
    Specifically, 4709 observations were missing information on one or more such variables, and these 4709 observations constitute the “missing data” file.
    The other 4618 observations which contained complete information were randomly split into an “analysis file” of 1540 observations and a “validation file” of 3078 observations.

    DATA 1978 contains 9327 individual records. Each individual record contains 28 columns of data, representing the following 19 variables.

    WHITE ALCHY JUNKY SUPER MARRIED FELON WORKREL PROPTY PERSON MALE PRIORS SCHOOL RULE AGE TSERVD FOLLOW RECID TIME FILE
    1 2 3 4 5 6 7 8 9 10 11-12 13-14 15-16 17-19 20-22 23-24 25-27 28

    WHITE is a dummy (indicator) variable equal to zero if the individual is black, and equal to one otherwise. Basically, WHITE equals one for whites and zero for blacks. However, the North Carolina prison population also contains a small number of Native Americans, Hispanics, Orientals, and individuals of “other” race. They are treated as whites, by the above definition.
    ALCHY is a dummy variable equal to one if the individual’s record indicates a serious problem with alcohol, and equal to zero otherwise. It is important to note that for individuals in the missing data sample (FILE = 3), the value of ALCHY is recorded as zero, but is meaningless.
    JUNKY is a dummy variable equal to one if the individual’s record indicates use of hard drugs, and equal to zero otherwise. It is important to note that for individuals in the missing data sample (FILE = 3), the value of JUNKY is recorded as zero, but is meaningless.
    SUPER is a dummy variable equal to one if the individual’s release from the sample sentence was supervised (e.g., parole), and equal to zero otherwise.
    MARRIED is a dummy variable equal to one if the individual was married at the time of release from the sample sentence, and equal to zero otherwise.
    FELON is a dummy variable equal to one if the sample conviction was for a felony, and equal to zero if it was for a misdemeanor.
    WORKREL is a dummy variable equal to one if the individual participated in the North Carolina prisoner work release program during the sample sentence, and equal to zero otherwise.
    PROPTY is a dummy variable equal to one if the sample conviction was for a crime against property, and equal to zero otherwise. A detailed listing of the crime codes which define this variable (and PERSON below) can be found in A. Witte, Work Release in North Carolina: An Evaluation of Its Post Release Effects, Chapel Hill, North Carolina: Institute for Research in Social Science.
    PERSON is a dummy variable equal to one if the sample conviction was for a crime against a person, and equal to zero otherwise. (Incidentally, note that PROPTY plus PERSON is not necessarily equal to one, because there is an additional miscellaneous category of offenses which are neither offenses against property nor offenses against a person.)
    MALE is a dummy variable equal to one if the individual is male, and equal to zero if the individual is female.
    PRIORS is the number of previous incarcerations, not including the sample sentence. The value -9 indicates that this information is missing.
    SCHOOL is the number of years of formal schooling completed. The value zero indicates that this information is missing.
    RULE is the number of prison rule violations reported during the sample sentence.
    AGE is age (in months) at time of release.
    TSERVD is the time served (in months) for the sample sentence.
    FOLLOW is the length of the followup period, in months. (The followup period is the time from relase until the North Carolina Department of Correction records were searched, in April, 1984.)
    RECID is a dummy variable equal to one if the individual returned to a North Carolina prison during the followup period, and equal to zero otherwise.
    TIME is the length of time from release from the sample sentence until return to prison in North Carolina, for individuals for whom RECID equals one. TIME is rounded to the nearest month. (In particular, note that TIME equals zero for individuals who return to prison in North Carolina within the first half month after release.) For individuals for whom RECID equals zero, the value of TIME is meaningless. For such individuals, TIME is usually recorded as zero, but it is occasionally recorded as the length of the followup period. We emphasize again that neither value is meaningful, for those individuals for whom RECID equals zero.
    FILE is a variable indicating to which data sample the individual record belongs. The value 1 indicates the analysis sample, 2 the validation sampel and 3 is missing data sample.
    ''')

    return(data_cont)

# readmission
def readmit(random_state=123, project_dir=None):
    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'readmit.csv.gz',
                    compression='gzip'),
    class_col = 'readmitted',
    project_dir = project_dir,
    save_dir = 'readmit',
    random_state=random_state,
    spiel = '''
    From Kaggle - https://www.kaggle.com/dansbecker/hospital-readmissions
    No further information
    ''')

    return(data_cont)

# readmission 0.1 sample
def readmit_samp(random_state=123, project_dir=None):
    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'readmit_samp.csv.gz',
                    compression='gzip'),
    class_col = 'readmitted',
    project_dir = project_dir,
    save_dir = 'readmit_samp',
    random_state=random_state,
    spiel = '''
    From Kaggle - https://www.kaggle.com/dansbecker/hospital-readmissions
    No further information
    0.1 of sample
    ''')

    return(data_cont)

# mental health survey 2014
def mhtech14(random_state=123, project_dir=None):
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'mhtech14.csv.gz',
                    compression='gzip')
    data.drop(columns='comments', inplace=True)
    var_names = data.columns.to_list()

    var_types = ['continuous',
                    'nominal',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'nominal',
                    'nominal']

    data_cont = data_container(
    data = data,
    class_col = 'treatment',
    var_types = var_types,
    project_dir = project_dir,
    save_dir = 'mhtech14',
    random_state=random_state,
    spiel = '''
    From Kaggle - https://www.kaggle.com/osmi/mental-health-in-tech-survey

    This dataset contains the following data:

    Timestamp
    Age
    Gender
    Country
    state: If you live in the United States, which state or territory do you live in?
    self_employed: Are you self-employed?
    family_history: Do you have a family history of mental illness?
    treatment: Have you sought treatment for a mental health condition?
    work_interfere: If you have a mental health condition, do you feel that it interferes with your work?
    no_employees: How many employees does your company or organization have?
    remote_work: Do you work remotely (outside of an office) at least 50% of the time?
    tech_company: Is your employer primarily a tech company/organization?
    benefits: Does your employer provide mental health benefits?
    care_options: Do you know the options for mental health care your employer provides?
    wellness_program: Has your employer ever discussed mental health as part of an employee wellness program?
    seek_help: Does your employer provide resources to learn more about mental health issues and how to seek help?
    anonymity: Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?
    leave: How easy is it for you to take medical leave for a mental health condition?
    mental_health_consequence: Do you think that discussing a mental health issue with your employer would have negative consequences?
    phys_health_consequence: Do you think that discussing a physical health issue with your employer would have negative consequences?
    coworkers: Would you be willing to discuss a mental health issue with your coworkers?
    supervisor: Would you be willing to discuss a mental health issue with your direct supervisor(s)?
    mental_health_interview: Would you bring up a mental health issue with a potential employer in an interview?
    phys_health_interview: Would you bring up a physical health issue with a potential employer in an interview?
    mental_vs_physical: Do you feel that your employer takes mental health as seriously as physical health?
    obs_consequence: Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?
    comments: Any additional notes or comments
    ''')

    return(data_cont)

# young people survey - smoking habit
def ypssmk(random_state=123, project_dir=None):
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'yps.csv.gz',
                    compression='gzip')
    var_names = [vn for vn in data.columns if vn != 'Smoking']
    var_names.append('Smoking')
    data = data[var_names]
    class_col = 'Smoking'
    data_cont = data_container(
    data = data,
    project_dir = project_dir,
    save_dir = 'ypssmk',
    random_state=random_state,
    spiel = '''
    https://www.kaggle.com/miroslavsabo/young-people-survey
    In 2013, students of the Statistics class at FSEV UK were asked to invite their friends to participate in this survey.

    The data file (responses.csv) consists of 1010 rows and 150 columns (139 integer and 11 categorical).
    For convenience, the original variable names were shortened in the data file. See the columns.csv file if you want to match the data with the original names.
    The data contain missing values.
    The survey was presented to participants in both electronic and written form.
    The original questionnaire was in Slovak language and was later translated into English.
    All participants were of Slovakian nationality, aged between 15-30.
    The variables can be split into the following groups:

    Music preferences (19 items)
    Movie preferences (12 items)
    Hobbies & interests (32 items)
    Phobias (10 items)
    Health habits (3 items)
    Personality traits, views on life, & opinions (57 items)
    Spending habits (7 items)
    Demographics (10 items)

    Questionnaire
    MUSIC PREFERENCES
    I enjoy listening to music.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I prefer.: Slow paced music 1-2-3-4-5 Fast paced music (integer)
    Dance, Disco, Funk: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Folk music: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Country: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Classical: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Musicals: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Pop: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Rock: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Metal, Hard rock: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Punk: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Hip hop, Rap: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Reggae, Ska: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Swing, Jazz: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Rock n Roll: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Alternative music: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Latin: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Techno, Trance: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Opera: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    MOVIE PREFERENCES
    I really enjoy watching movies.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    Horror movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Thriller movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Comedies: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Romantic movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Sci-fi movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    War movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Tales: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Cartoons: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Documentaries: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Western movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Action movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    HOBBIES & INTERESTS
    History: Not interested 1-2-3-4-5 Very interested (integer)
    Psychology: Not interested 1-2-3-4-5 Very interested (integer)
    Politics: Not interested 1-2-3-4-5 Very interested (integer)
    Mathematics: Not interested 1-2-3-4-5 Very interested (integer)
    Physics: Not interested 1-2-3-4-5 Very interested (integer)
    Internet: Not interested 1-2-3-4-5 Very interested (integer)
    PC Software, Hardware: Not interested 1-2-3-4-5 Very interested (integer)
    Economy, Management: Not interested 1-2-3-4-5 Very interested (integer)
    Biology: Not interested 1-2-3-4-5 Very interested (integer)
    Chemistry: Not interested 1-2-3-4-5 Very interested (integer)
    Poetry reading: Not interested 1-2-3-4-5 Very interested (integer)
    Geography: Not interested 1-2-3-4-5 Very interested (integer)
    Foreign languages: Not interested 1-2-3-4-5 Very interested (integer)
    Medicine: Not interested 1-2-3-4-5 Very interested (integer)
    Law: Not interested 1-2-3-4-5 Very interested (integer)
    Cars: Not interested 1-2-3-4-5 Very interested (integer)
    Art: Not interested 1-2-3-4-5 Very interested (integer)
    Religion: Not interested 1-2-3-4-5 Very interested (integer)
    Outdoor activities: Not interested 1-2-3-4-5 Very interested (integer)
    Dancing: Not interested 1-2-3-4-5 Very interested (integer)
    Playing musical instruments: Not interested 1-2-3-4-5 Very interested (integer)
    Poetry writing: Not interested 1-2-3-4-5 Very interested (integer)
    Sport and leisure activities: Not interested 1-2-3-4-5 Very interested (integer)
    Sport at competitive level: Not interested 1-2-3-4-5 Very interested (integer)
    Gardening: Not interested 1-2-3-4-5 Very interested (integer)
    Celebrity lifestyle: Not interested 1-2-3-4-5 Very interested (integer)
    Shopping: Not interested 1-2-3-4-5 Very interested (integer)
    Science and technology: Not interested 1-2-3-4-5 Very interested (integer)
    Theatre: Not interested 1-2-3-4-5 Very interested (integer)
    Socializing: Not interested 1-2-3-4-5 Very interested (integer)
    Adrenaline sports: Not interested 1-2-3-4-5 Very interested (integer)
    Pets: Not interested 1-2-3-4-5 Very interested (integer)
    PHOBIAS
    Flying: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    Thunder, lightning: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    Darkness: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    Heights: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    Spiders: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    Snakes: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    Rats, mice: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    Ageing: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    Dangerous dogs: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    Public speaking: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    HEALTH HABITS
    Smoking habits: Never smoked - Tried smoking - Former smoker - Current smoker (categorical)
    Drinking: Never - Social drinker - Drink a lot (categorical)
    I live a very healthy lifestyle.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    PERSONALITY TRAITS, VIEWS ON LIFE & OPINIONS
    I take notice of what goes on around me.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I try to do tasks as soon as possible and not leave them until last minute.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I always make a list so I don't forget anything.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I often study or work even in my spare time.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I look at things from all different angles before I go ahead.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I believe that bad people will suffer one day and good people will be rewarded.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I am reliable at work and always complete all tasks given to me.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I always keep my promises.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I can fall for someone very quickly and then completely lose interest.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I would rather have lots of friends than lots of money.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I always try to be the funniest one.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I can be two faced sometimes.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I damaged things in the past when angry.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I take my time to make decisions.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I always try to vote in elections.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I often think about and regret the decisions I make.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I can tell if people listen to me or not when I talk to them.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I am a hypochondriac.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I am emphatetic person.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I eat because I have to. I don't enjoy food and eat as fast as I can.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I try to give as much as I can to other people at Christmas.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I don't like seeing animals suffering.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I look after things I have borrowed from others.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I feel lonely in life.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I used to cheat at school.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I worry about my health.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I wish I could change the past because of the things I have done.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I believe in God.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I always have good dreams.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I always give to charity.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I have lots of friends.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    Timekeeping.: I am often early. - I am always on time. - I am often running late. (categorical)
    Do you lie to others?: Never. - Only to avoid hurting someone. - Sometimes. - Everytime it suits me. (categorical)
    I am very patient.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I can quickly adapt to a new environment.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    My moods change quickly.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I am well mannered and I look after my appearance.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I enjoy meeting new people.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I always let other people know about my achievements.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I think carefully before answering any important letters.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I enjoy childrens' company.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I am not afraid to give my opinion if I feel strongly about something.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I can get angry very easily.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I always make sure I connect with the right people.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I have to be well prepared before public speaking.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I will find a fault in myself if people don't like me.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I cry when I feel down or things don't go the right way.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I am 100% happy with my life.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I am always full of life and energy.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I prefer big dangerous dogs to smaller, calmer dogs.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I believe all my personality traits are positive.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    If I find something the doesn't belong to me I will hand it in.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I find it very difficult to get up in the morning.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I have many different hobbies and interests.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I always listen to my parents' advice.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I enjoy taking part in surveys.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    How much time do you spend online?: No time at all - Less than an hour a day - Few hours a day - Most of the day (categorical)
    SPENDING HABITS
    I save all the money I can.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I enjoy going to large shopping centres.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I prefer branded clothing to non branded.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I spend a lot of money on partying and socializing.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I spend a lot of money on my appearance.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I spend a lot of money on gadgets.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I will hapilly pay more money for good, quality or healthy food.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    DEMOGRAPHICS
    Age: (integer)
    Height: (integer)
    Weight: (integer)
    How many siblings do you have?: (integer)
    Gender: Female - Male (categorical)
    I am: Left handed - Right handed (categorical)
    Highest education achieved: Currently a Primary school pupil - Primary school - Secondary school - College/Bachelor degree (categorical)
    I am the only child: No - Yes (categorical)
    I spent most of my childhood in a: City - village (categorical)
    I lived most of my childhood in a: house/bungalow - block of flats (categorical)
    ''')

# young people survey - alcohol habit
def ypsalc(random_state=123, project_dir=None):
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'yps.csv.gz',
                    compression='gzip')
    var_names = [vn for vn in data.columns if vn != 'Alcohol']
    var_names.append('Alcohol')
    data = data[var_names]
    class_col = 'Alcohol'
    data_cont = data_container(
    data = data,
    project_dir = project_dir,
    save_dir = 'ypsalc',
    random_state=random_state,
    spiel = '''
    https://www.kaggle.com/miroslavsabo/young-people-survey
    In 2013, students of the Statistics class at FSEV UK were asked to invite their friends to participate in this survey.

    The data file (responses.csv) consists of 1010 rows and 150 columns (139 integer and 11 categorical).
    For convenience, the original variable names were shortened in the data file. See the columns.csv file if you want to match the data with the original names.
    The data contain missing values.
    The survey was presented to participants in both electronic and written form.
    The original questionnaire was in Slovak language and was later translated into English.
    All participants were of Slovakian nationality, aged between 15-30.
    The variables can be split into the following groups:

    Music preferences (19 items)
    Movie preferences (12 items)
    Hobbies & interests (32 items)
    Phobias (10 items)
    Health habits (3 items)
    Personality traits, views on life, & opinions (57 items)
    Spending habits (7 items)
    Demographics (10 items)

    Questionnaire
    MUSIC PREFERENCES
    I enjoy listening to music.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I prefer.: Slow paced music 1-2-3-4-5 Fast paced music (integer)
    Dance, Disco, Funk: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Folk music: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Country: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Classical: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Musicals: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Pop: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Rock: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Metal, Hard rock: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Punk: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Hip hop, Rap: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Reggae, Ska: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Swing, Jazz: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Rock n Roll: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Alternative music: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Latin: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Techno, Trance: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Opera: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    MOVIE PREFERENCES
    I really enjoy watching movies.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    Horror movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Thriller movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Comedies: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Romantic movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Sci-fi movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    War movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Tales: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Cartoons: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Documentaries: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Western movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    Action movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much (integer)
    HOBBIES & INTERESTS
    History: Not interested 1-2-3-4-5 Very interested (integer)
    Psychology: Not interested 1-2-3-4-5 Very interested (integer)
    Politics: Not interested 1-2-3-4-5 Very interested (integer)
    Mathematics: Not interested 1-2-3-4-5 Very interested (integer)
    Physics: Not interested 1-2-3-4-5 Very interested (integer)
    Internet: Not interested 1-2-3-4-5 Very interested (integer)
    PC Software, Hardware: Not interested 1-2-3-4-5 Very interested (integer)
    Economy, Management: Not interested 1-2-3-4-5 Very interested (integer)
    Biology: Not interested 1-2-3-4-5 Very interested (integer)
    Chemistry: Not interested 1-2-3-4-5 Very interested (integer)
    Poetry reading: Not interested 1-2-3-4-5 Very interested (integer)
    Geography: Not interested 1-2-3-4-5 Very interested (integer)
    Foreign languages: Not interested 1-2-3-4-5 Very interested (integer)
    Medicine: Not interested 1-2-3-4-5 Very interested (integer)
    Law: Not interested 1-2-3-4-5 Very interested (integer)
    Cars: Not interested 1-2-3-4-5 Very interested (integer)
    Art: Not interested 1-2-3-4-5 Very interested (integer)
    Religion: Not interested 1-2-3-4-5 Very interested (integer)
    Outdoor activities: Not interested 1-2-3-4-5 Very interested (integer)
    Dancing: Not interested 1-2-3-4-5 Very interested (integer)
    Playing musical instruments: Not interested 1-2-3-4-5 Very interested (integer)
    Poetry writing: Not interested 1-2-3-4-5 Very interested (integer)
    Sport and leisure activities: Not interested 1-2-3-4-5 Very interested (integer)
    Sport at competitive level: Not interested 1-2-3-4-5 Very interested (integer)
    Gardening: Not interested 1-2-3-4-5 Very interested (integer)
    Celebrity lifestyle: Not interested 1-2-3-4-5 Very interested (integer)
    Shopping: Not interested 1-2-3-4-5 Very interested (integer)
    Science and technology: Not interested 1-2-3-4-5 Very interested (integer)
    Theatre: Not interested 1-2-3-4-5 Very interested (integer)
    Socializing: Not interested 1-2-3-4-5 Very interested (integer)
    Adrenaline sports: Not interested 1-2-3-4-5 Very interested (integer)
    Pets: Not interested 1-2-3-4-5 Very interested (integer)
    PHOBIAS
    Flying: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    Thunder, lightning: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    Darkness: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    Heights: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    Spiders: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    Snakes: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    Rats, mice: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    Ageing: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    Dangerous dogs: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    Public speaking: Not afraid at all 1-2-3-4-5 Very afraid of (integer)
    HEALTH HABITS
    Smoking habits: Never smoked - Tried smoking - Former smoker - Current smoker (categorical)
    Drinking: Never - Social drinker - Drink a lot (categorical)
    I live a very healthy lifestyle.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    PERSONALITY TRAITS, VIEWS ON LIFE & OPINIONS
    I take notice of what goes on around me.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I try to do tasks as soon as possible and not leave them until last minute.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I always make a list so I don't forget anything.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I often study or work even in my spare time.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I look at things from all different angles before I go ahead.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I believe that bad people will suffer one day and good people will be rewarded.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I am reliable at work and always complete all tasks given to me.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I always keep my promises.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I can fall for someone very quickly and then completely lose interest.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I would rather have lots of friends than lots of money.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I always try to be the funniest one.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I can be two faced sometimes.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I damaged things in the past when angry.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I take my time to make decisions.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I always try to vote in elections.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I often think about and regret the decisions I make.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I can tell if people listen to me or not when I talk to them.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I am a hypochondriac.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I am emphatetic person.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I eat because I have to. I don't enjoy food and eat as fast as I can.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I try to give as much as I can to other people at Christmas.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I don't like seeing animals suffering.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I look after things I have borrowed from others.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I feel lonely in life.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I used to cheat at school.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I worry about my health.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I wish I could change the past because of the things I have done.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I believe in God.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I always have good dreams.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I always give to charity.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I have lots of friends.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    Timekeeping.: I am often early. - I am always on time. - I am often running late. (categorical)
    Do you lie to others?: Never. - Only to avoid hurting someone. - Sometimes. - Everytime it suits me. (categorical)
    I am very patient.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I can quickly adapt to a new environment.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    My moods change quickly.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I am well mannered and I look after my appearance.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I enjoy meeting new people.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I always let other people know about my achievements.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I think carefully before answering any important letters.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I enjoy childrens' company.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I am not afraid to give my opinion if I feel strongly about something.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I can get angry very easily.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I always make sure I connect with the right people.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I have to be well prepared before public speaking.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I will find a fault in myself if people don't like me.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I cry when I feel down or things don't go the right way.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I am 100% happy with my life.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I am always full of life and energy.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I prefer big dangerous dogs to smaller, calmer dogs.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I believe all my personality traits are positive.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    If I find something the doesn't belong to me I will hand it in.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I find it very difficult to get up in the morning.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I have many different hobbies and interests.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I always listen to my parents' advice.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I enjoy taking part in surveys.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    How much time do you spend online?: No time at all - Less than an hour a day - Few hours a day - Most of the day (categorical)
    SPENDING HABITS
    I save all the money I can.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I enjoy going to large shopping centres.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I prefer branded clothing to non branded.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I spend a lot of money on partying and socializing.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I spend a lot of money on my appearance.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I spend a lot of money on gadgets.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    I will hapilly pay more money for good, quality or healthy food.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)
    DEMOGRAPHICS
    Age: (integer)
    Height: (integer)
    Weight: (integer)
    How many siblings do you have?: (integer)
    Gender: Female - Male (categorical)
    I am: Left handed - Right handed (categorical)
    Highest education achieved: Currently a Primary school pupil - Primary school - Secondary school - College/Bachelor degree (categorical)
    I am the only child: No - Yes (categorical)
    I spent most of my childhood in a: City - village (categorical)
    I lived most of my childhood in a: house/bungalow - block of flats (categorical)
    ''')

# readmission
def noshow(random_state=123, project_dir=None):
    data_cont = data_container(
    data = pd.read_csv('CHIRPS' + cfg.path_sep + 'datafiles' + cfg.path_sep + 'noshow.csv.gz',
                    compression='gzip'),
    class_col = 'readmitted',
    project_dir = project_dir,
    save_dir = 'readmit',
    random_state=random_state,
    spiel = '''
    From Kaggle - https://www.kaggle.com/dansbecker/hospital-readmissions
    No further information
    ''')

    return(data_cont)
