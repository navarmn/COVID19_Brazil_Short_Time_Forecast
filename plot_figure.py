import os
from os import path

import numpy as np
import pandas as pd

import collections

import utils

# import logging as log
# log.basicConfig(level=log.DEBUG)

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta


DATA_FOLDER = path.join('..', 'data')
RESULTS_FOLDER = path.join('..', 'results')

STATES = ['AC', 'AL', 'AM', 'AP', 'BA', 'CE',
         'DF', 'ES', 'GO', 'MA', 'MT', 'MS',
         'MG', 'PA', 'PB', 'PR', 'PE', 'PI',
         'RJ', 'RN', 'RO', 'RS', 'RR', 'SC',
         'SE', 'SP', 'TO']

MODELS = ['linear-regression', 'gaussian-process', 'mlp', 'svr', 'exponential-holt']

STRATEGIES = [
    'aggregated',
    'windowed'
]

METRICS = ['MAE', 'RMSE', 'MSLE']


def get_date(state, to_drop=10, strategy='aggregated'):
    if strategy == 'aggregated':
        df_state = utils.download_state(state=state)
        df_state = df_state[df_state['cases'] != 0]
        df_state = df_state.iloc[:-to_drop]
    if strategy == 'windowed':
        df_state = utils.download_state(state=state)
        df_state = df_state[df_state['cases'] != 0]
        df_state = df_state.iloc[:-to_drop]

    return df_state




dayone = df_state.index[0]
    dayout = datetime.strptime(df_state.index[-1], '%d/%m/%Y')  
    days = np.array(utils.count_days(dayone=dayone, dayout=dayout, date_string='%d/%m/%Y'))
    X = days.reshape(-1,1)
    y = utils.get_labels(df_state['cases']).reshape(-1,1)
    X, y = utils.check_inputs(X, y)