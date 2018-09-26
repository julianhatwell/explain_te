import io
import zipfile
import urllib
import numpy as np
import pandas as pd
from datetime import datetime
import math
import julian

# adult from source
if True:
    '''
    random_state = 123
    var_names = ['age'
           , 'workclass'
           , 'lfnlwgt'
           , 'education'
           , 'educationnum'
           , 'maritalstatus'
           , 'occupation'
           , 'relationship'
           , 'race'
           , 'sex'
           , 'lcapitalgain'
           , 'lcapitalloss'
           , 'hoursperweek'
           , 'nativecountry'
           , 'income']

    vars_types = ['continuous'
           , 'nominal'
           , 'continuous'
           , 'nominal'
           , 'continuous'
           , 'nominal'
           , 'nominal'
           , 'nominal'
           , 'nominal'
           , 'nominal'
           , 'continuous'
           , 'continuous'
           , 'continuous'
           , 'nominal'
           , 'nominal']

    adult_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    adult_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

    adult_train_bytes = urllib.request.urlopen(target_url)
    adult_test_bytes = urllib.request.urlopen(target_url)
    adult_train = pd.read_csv(adult_bytes, header=None, names=var_names)
    adult_test = pd.read_csv(adult_test_bytes, header=None, names=var_names)

    # combine the two datasets and split them later with standard code
    frames = [adult_train, adult_test]
    adult = pd.concat(frames)

    # some tidying required
    adult.income = adult.income.str.replace('.', '')
    for f, t in zip(var_names, vars_types):
        if t == 'continuous':
            adult[f] = adult[f].astype('int32')
        else:
            adult[f] = adult[f].str.replace(' ', '')
    # change question mark character to 'Unknown'
    qm_to_unk = lambda w: 'Unknown' if w == '?' else w
    # duplication / tidy of a country entry
    tt_fix = lambda w: 'Trinidad and Tobago' if w == 'Trinadad&Tobago' else w

    adult['workclass'] = adult.workclass.apply(qm_to_unk)
    adult['nativecountry'] = adult.nativecountry.apply(qm_to_unk)
    adult['nativecountry'] = adult.nativecountry.apply(tt_fix)

    # make these numeric values a bit closer to normal distr.
    adult['lcapitalgain'] = np.log(adult['lcapitalgain'] + abs(adult['lcapitalgain'].min()) + 1)
    adult['lcapitalloss'] = np.log(adult['lcapitalloss'] + abs(adult['lcapitalloss'].min()) + 1)
    adult['lfnlwgt'] = np.log(adult['lfnlwgt'] + abs(adult['lfnlwgt'].min()) + 1)

    # create a small set that is easier to play with on a laptop
    adult_samp = adult.sample(frac=0.5, random_state=random_state).reset_index()
    adult_samp.drop(labels='index', axis=1, inplace=True)

    # create a small set that is easier to play with on a laptop
    adult_small_samp = adult.sample(frac=0.05, random_state=random_state).reset_index()
    adult_small_samp.drop(labels='index', axis=1, inplace=True)

    # save
    adult.to_csv('forest_surveyor\\datafiles\\adult.csv.gz', index=False, compression='gzip')
    adult_samp.to_csv('forest_surveyor\\datafiles\\adult_samp.csv.gz', index=False, compression='gzip')
    adult_small_samp.to_csv('forest_surveyor\\datafiles\\adult_small_samp.csv.gz', index=False, compression='gzip')
    '''

# bankmark from source
if True:
    '''
    # random seed for train test split and sampling
    random_state = 123

    def import_bankmark_file(file):
        archive = zipfile.ZipFile('forest_surveyor/source_datafiles/bank-additional-full.zip', 'r')
        lines = archive.read(file).decode("utf-8").split('\r\n')
        lines = [lines[i].replace('\"', '').split(';') for i in range(1, len(lines))]
        lines.pop() # last item is empty
        archive.close()
        return(lines)

    test_lines = import_bankmark_file('bank-additional.csv')
    train_lines = import_bankmark_file('bank-additional-full.csv')
    all_lines = train_lines + test_lines

    names = ['age','job','marital','education','default','housing','loan','contact','month',
             'day_of_week','duration','campaign','pdays','previous','poutcome','emp.var.rate',
             'cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y']

    vtypes = {'age' : np.uint8, 'job' : object, 'marital' : object, 'education' : object, 'default' : object,
                  'housing' : object, 'loan' : object, 'contact' : object, 'month' : object, 'day_of_week' : object,
                  'duration' : np.uint16, 'campaign' : np.uint8, 'pdays' : np.uint8, 'previous' : np.uint8,
                  'poutcome' : object, 'emp.var.rate' : np.float16, 'cons.price.idx' : np.float16,
                  'cons.conf.idx' : np.float16, 'euribor3m' : np.float16, 'nr.employed' : np.float16, 'y' : object}

    bankmark = pd.DataFrame(all_lines, columns=names)
    bankmark = bankmark_raw.astype(dtype=vtypes)

    # save
    bankmark.to_csv('forest_surveyor\\datafiles\\bankmark.csv.gz', index=False, compression='gzip')

    # create small set that is easier to play with on a laptop
    samp = bankmark.sample(frac=0.05, random_state=random_state).reset_index()
    samp.drop(labels='index', axis=1, inplace=True)
    samp.to_csv('forest_surveyor\\datafiles\\bankmark_samp.csv.gz', index=False, compression='gzip')
    '''

# car form source
if True:
    '''
    var_names = ['buying'
                , 'maint'
                , 'doors'
                , 'persons'
                , 'lug_boot'
                , 'safety'
                , 'acceptability']

    target_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'

    car_bytes = urllib.request.urlopen(target_url)
    car = pd.read_csv(car_bytes, header=None, names=var_names)
    # recode to a 2 class subproblems
    car.acceptability.loc[car.acceptability != 'unacc'] = 'acc'

    car.to_csv('forest_surveyor\\datafiles\\car.csv.gz'), index=False, compression='gzip')
    '''

# cardio
if True:
    '''
    target_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls'
    cardio_bytes = urllib.request.urlopen(target_url)
    cardio = pd.read_excel(cardio_bytes,
                            sheet_name='Data',
                            header=1)

    var_names_raw = ['LB'
                , 'AC.1'
                , 'FM.1'
                , 'UC.1'
                , 'DL.1'
                , 'DS.1'
                , 'DP.1'
                , 'ASTV'
                , 'MSTV'
                , 'ALTV'
                , 'MLTV'
                , 'Width'
                , 'Min'
                , 'Max'
                , 'Nmax'
                , 'Nzeros'
                , 'Mode'
                , 'Mean'
                , 'Median'
                , 'Variance'
                , 'Tendency'
                #, 'CLASS' Exclude as this is an alternative target
                , 'NSP']

    cardio = cardio.loc[:, var_names_raw]

    var_names = ['LB'
                , 'AC'
                , 'FM'
                , 'UC'
                , 'DL'
                , 'DS'
                , 'DP'
                , 'ASTV'
                , 'MSTV'
                , 'ALTV'
                , 'MLTV'
                , 'Width'
                , 'Min'
                , 'Max'
                , 'Nmax'
                , 'Nzeros'
                , 'Mode'
                , 'Mean'
                , 'Median'
                , 'Variance'
                , 'Tendency'
                #, 'CLASS'
                , 'NSP']

    cardio.columns = var_names

    # remove the last three rows that are aggragates in the raw data file
    cardio = cardio.loc[~cardio['LB'].isna(), :]

    # re-code NSP and delete class variable
    NSP = pd.Series(['N'] * cardio.shape[0])
    NSP.loc[cardio.NSP.values == 2] = 'S'
    NSP.loc[cardio.NSP.values == 3] = 'P'
    cardio.NSP = NSP

    # save
    cardio.to_csv('forest_surveyor\\datafiles\\cardio.csv.gz', index=False, compression='gzip')
    '''

# credit from source
if True:
    '''
    var_names = ['A1'
                , 'A2'
                , 'A3'
                , 'A4'
                , 'A5'
                , 'A6'
                , 'A7'
                , 'A8'
                , 'A9'
                , 'A10'
                , 'A11'
                , 'A12'
                , 'A13'
                , 'A14'
                , 'A15'
                , 'A16']

    vars_types = ['nominal'
        , 'continuous'
        , 'continuous'
        , 'nominal'
        , 'nominal'
        , 'nominal'
        , 'nominal'
        , 'continuous'
        , 'nominal'
        , 'nominal'
        , 'continuous'
        , 'nominal'
        , 'nominal'
        , 'continuous'
        , 'continuous'
        , 'nominal'
    ]

    target_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'

    credit_bytes = urllib.request.urlopen(target_url)
    credit = pd.read_csv(credit_bytes,
                         header=None,
                         delimiter=',',
                         index_col=False,
                         names=var_names,
                         na_values = '?')

    # re-code rating class variable
    A16 = pd.Series(['plus'] * credit.shape[0])
    A16.loc[credit.A16.values == '-'] = 'minus'
    credit.A16 = A16

    # deal with some missing data
    for v, t in zip(var_names, vars_types):
        if t == 'nominal':
            credit[v] = credit[v].fillna('u')
        else:
            credit[v] = credit[v].fillna(credit[v].mean())

    credit.to_csv('forest_surveyor\\datafiles\\credit.csv.gz', index=False, compression='gzip')
    '''

# german from source
if True:
    '''
    var_names = ['chk'
                , 'dur'
                , 'crhis'
                , 'pps'
                , 'amt'
                , 'svng'
                , 'emp'
                , 'rate'
                , 'pers'
                , 'debt'
                , 'res'
                , 'prop'
                , 'age'
                , 'plans'
                , 'hous'
                , 'creds'
                , 'job'
                , 'deps'
                , 'tel'
                , 'foreign'
                , 'rating']

    target_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'

    german_bytes = urllib.request.urlopen(target_url)
    german = pd.read_csv(german_bytes,
                         header=None,
                         delimiter=' ',
                         index_col=False,
                         names=var_names)

    # re-code rating class variable
    rating = pd.Series(['good'] * german.count()[0])
    rating.loc[german.rating == 2] = 'bad'
    german.rating = rating

    german.to_csv('forest_surveyor\\datafiles\\german.csv.gz', index=False, compression='gzip')
    '''

# lending from source
if True:
    '''
    # lending from source
    # download the files accepted_2007_to_2018Q2.csv.gz from Kaggle
    # https://www.kaggle.com/wendykan/lending-club-loan-data
    # there is also a rejected_2007_to_2018Q2.csv.gz file but we are interested in who paid off a loan
    random_state=123
    lending = pd.read_csv('forest_surveyor\\source_datafiles\\accepted_2007_to_2018Q2.csv.gz',
                          compression='gzip', low_memory=False)     # low_memory=False prevents mixed data types in the DataFrame

    # Just looking at loans that met the policy and were either fully paid or charged off (finally defaulted)
    lending = lending.loc[lending['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    lending.reset_index(inplace=True, drop=True)

    # data set is wide. What can be done to reduce it? lots to clean up and some useful transforms

    # drop cols with only one distinct value
    drop_list = []
    for col in lending.columns:
        if lending[col].nunique() == 1:
            drop_list.append(col)

    lending.drop(labels=drop_list, axis=1, inplace=True)

    # drop cols with excessively high missing amounts
    drop_list = []
    for col in lending.columns:
        if lending[col].notnull().sum() / lending.shape[0] < 0.5:
            drop_list.append(col)

    lending.drop(labels=drop_list, axis=1, inplace=True)

    # more noisy columns
    lending.drop(labels=['id', 'title', 'emp_title', 'url',
                         'application_type', 'acc_now_delinq',
                         'num_tl_120dpd_2m', 'num_tl_30dpd'],
                 axis=1, inplace=True)

    # highly correlated with the class
    lending.drop(labels=['collection_recovery_fee', 'debt_settlement_flag', 'recoveries'], axis=1, inplace=True)

    # no need for an upper and lower fico, they are perfectly correlated. Take the mean of each pair.
    fic = ['fico_range_low', 'fico_range_high']
    lastfic = ['last_fico_range_low', 'last_fico_range_high']
    lending['fico'] = lending[fic].mean(axis=1)
    lending['last_fico'] = lending[lastfic].mean(axis=1)
    lending.drop(labels=fic + lastfic, axis=1, inplace=True)

    # slightly more informative coding of these vars that are mostly correlated with loan amnt and/or high skew
    lending['non_funded_score'] = np.log(lending['loan_amnt'] + 1 - lending['funded_amnt'])
    lending['non_funded_inv_score'] = np.log(lending['loan_amnt'] + 1 - lending['funded_amnt_inv'])
    lending['adj_log_dti'] = np.log(lending['dti'] + abs(lending['dti'].min()) + 1)
    lending['log_inc'] = np.log(lending['annual_inc'] + abs(lending['annual_inc'].min()) + 1)
    lending.drop(['funded_amnt', 'funded_amnt_inv', 'dti', 'annual_inc'], axis=1, inplace=True)

    # julian dates are better, nice continuous input.
    jul_conv = lambda x : np.nan if x == 'Unknown' else julian.to_jd(datetime.strptime(x, '%b-%Y'))
    for date_col in ['issue_d' , 'last_credit_pull_d', 'earliest_cr_line', 'last_pymnt_d']:
        dc = pd.Series(['Unknown'] * lending.shape[0])
        dc.loc[~lending[date_col].isnull().values] = lending[date_col].loc[~lending[date_col].isnull().values]
        lending[date_col] = dc.map(jul_conv)

    # this one feature has just a tiny number of missing. OK to impute.
    lending['last_credit_pull_d'] = lending.last_credit_pull_d.fillna(lending.last_credit_pull_d.mean())

    # convert 'term' to int
    lending['term'] = lending['term'].apply(lambda s:np.float(s[1:3])) # There's an extra space in the data for some reason

    # convert sub-grade to float and remove grade
    grade_dict = {'A':0.0, 'B':1.0, 'C':2.0, 'D':3.0, 'E':4.0, 'F':5.0, 'G':6.0}
    grade_to_float = lambda s : 5 * grade_dict[s[0]] + np.float(s[1]) - 1

    lending['sub_grade'] = lending['sub_grade'].map(grade_to_float)
    lending.drop(labels='grade', axis=1, inplace=True)

    # convert emp_length - assume missing and < 0 is no job or only very recent started job
    # emp length is only significant for values of 0 or not 0
    emp_conv = lambda e : 'U' if pd.isnull(e) or e[0] == '<' else 'E'
    lending['emp'] = lending['emp_length'].map(emp_conv)
    lending.drop(labels='emp_length', axis=1, inplace=True)

    # tidy up some very minor class codes in home ownership
    lending['home_ownership'] = lending['home_ownership'].map(lambda h: 'OTHER' if h in ['ANY', 'NONE'] else h)

    # there is a number of rows that have missing data for many variables in a block pattern -
    # these are probably useless because missingness goes across so many variables
    # it might be possible to save them to a different set and create a separate model on them

    # another approach is to fill them with an arbitrary data point (means, zeros, whatever)
    # and add a new feature that is binary for whether this row had missing data
    # this will give the model something to adjust/correlate/associate with if these rows turn out to add noise

    # 'avg_cur_bal is a template for block missingness
    # will introduce a missing indicator column based on this
    # then fillna with zeros and finally filter out some unsalvageable really rows
    lending['block_missingness'] = lending['avg_cur_bal'].isnull() * 1.0
    # rows where last_pymnt_d is zero are just a mess, get them outa here. all other nans get changed to zero
    lending = lending.fillna(0)
    lending = lending[lending.last_pymnt_d != 0]
    # and a final reindex
    lending.reset_index(inplace=True, drop=True)

    # and rearrange so class_col is at the end
    class_col = 'loan_status'
    pos = np.where(lending.columns == class_col)[0][0]
    var_names = list(lending.columns[:pos]) + list(lending.columns[pos + 1:]) + list(lending.columns[pos:pos + 1])
    lending = lending[var_names]

    # create a small set that is easier to play with on a laptop
    lend_samp = lending.sample(frac=0.1, random_state=random_state).reset_index()
    lend_samp.drop(labels='index', axis=1, inplace=True)
    lend_small_samp = lending.sample(frac=0.01, random_state=random_state).reset_index()
    lend_small_samp.drop(labels='index', axis=1, inplace=True)
    lend_tiny_samp = lending.sample(frac=0.0025, random_state=random_state).reset_index()
    lend_tiny_samp.drop(labels='index', axis=1, inplace=True)

    # save
    lending.to_csv('forest_surveyor\\datafiles\\lending.csv.gz', index=False, compression='gzip')
    lend_samp.to_csv('forest_surveyor\\datafiles\\lend_samp.csv.gz', index=False, compression='gzip')
    lend_small_samp.to_csv('forest_surveyor\\datafiles\\lend_small_samp.csv.gz', index=False, compression='gzip')
    lend_tiny_samp.to_csv('forest_surveyor\\datafiles\\lend_tiny_samp.csv.gz', index=False, compression='gzip')
    '''

# nursery
if True:
    '''
    var_names = ['parents'
               , 'has_nurs'
               , 'form'
               , 'children'
               , 'housing'
               , 'finance'
               , 'social'
               , 'health'
               , 'decision']

    vars_types = ['nominal'
               , 'nominal'
               , 'nominal'
               , 'nominal'
               , 'nominal'
               , 'nominal'
               , 'nominal'
               , 'nominal'
               , 'nominal']

    nursery = pd.read_csv(pickle_path('nursery.csv')
                          , names=var_names)

    # filter one row where class == 2
    nursery = nursery[nursery.decision != 'recommend']
    # reset the pandas index
    nursery.index = range(len(nursery))

    nursery.to_csv(pickle_path('nursery.csv.gz'), index=False, compression='gzip')
    '''

# rcdv
if True:
    '''
    # random seed for train test split and sampling
    random_state = 123
    rcdv = pd.read_excel('data_source_files\\rcdv.xlsx'
                                        , sheet_name='1978'
                                        , header=0)
    # merging two sheets brought in row numbers
    rcdv = rcdv.append(pd.read_excel('data_source_files\\rcdv.xlsx'
                                    , sheet_name='1980'
                                    , header=0))

    # tidy index and drop col1
    rcdv.reset_index(drop=True, inplace=True)
    rcdv.drop(labels='Column1', axis=1, inplace=True)

    # rename file to miss as it is needed to indicate where were missing values
    rcdv.columns = ['miss' if vn == 'file' else vn for vn in rcdv.columns]

    var_names=list(rcdv)[:16] + list(rcdv)[17:] + list(rcdv)[16:17] # put recid to the end
    rcdv = rcdv[var_names]

    vars_types = ['nominal'
                    , 'nominal'
                    , 'nominal'
                    , 'nominal'
                    , 'nominal'
                    , 'nominal'
                    , 'nominal'
                    , 'nominal'
                    , 'nominal'
                    , 'nominal'
                    , 'continuous'
                    , 'continuous'
                    , 'continuous'
                    , 'continuous'
                    , 'continuous'
                    , 'continuous'
                    , 'nominal'
                    , 'continuous'
                    , 'nominal'
                    , 'nominal']

    class_col = 'recid'
    features = [vn for vn in var_names if vn != class_col]

    # recode priors, all that were set to -9 were missing, and it is logged in the file variable (3 = missing data indicator)
    rcdv['priors'] = rcdv['priors'].apply(lambda x: 0 if x == -9 else x)
    rcdv['miss'] = rcdv['miss'].apply(lambda x: 1 if x == 3 else 0)
    rcdv['recid'] = rcdv['recid'].apply(lambda x: 'T' if x == 1 else 'F')

    # remove cols we don't want. Time is only useful in survival analysis. Correlates exactly with recid.
    to_be_del = ['time']
    for tbd in to_be_del:
        del rcdv[tbd]
        del vars_types[np.where(np.array(var_names) == tbd)[0][0]]
        del var_names[np.where(np.array(var_names) == tbd)[0][0]]
        del features[np.where(np.array(features) == tbd)[0][0]]

    # save it out
    rcdv.to_csv('forest_surveyor\\datafiles\\rcdv.csv.gz', index=False, compression='gzip')

    # create small set that is easier to play with on a laptop
    samp = rcdv.sample(frac=0.1, random_state=random_state)
    samp.reset_index(drop=True, inplace=True)
    samp.to_csv('forest_surveyor\\datafiles\\rcdv_samp.csv.gz', index=False, compression='gzip')
    '''
