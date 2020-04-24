import os
from os import path
import sys
sys.path.append('..')

import numpy as np
import pandas as pd

import collections

import utils

import logging as log
log.basicConfig(level=log.DEBUG)


DATA_FOLDER = path.join('..', 'data')
RESULTS_FOLDER = path.join('..', 'results')

STATES = ['AC', 'AL', 'AM', 'AP', 'BA', 'CE',
         'DF', 'ES', 'GO', 'MA', 'MT', 'MS',
         'MG', 'PA', 'PB', 'PR', 'PE', 'PI',
         'RJ', 'RN', 'RO', 'RS', 'RR', 'SC',
         'SE', 'SP', 'TO']

MODELS = ['linear-regression', 'exponential-holt']

STRATEGIES = [
    'aggregated',
    'windowed'
]

METRICS = ['MAE', 'RMSE', 'MSLE']


df_state = {}
for state in STATES:
    df_state[state] = pd.read_csv(path.join(DATA_FOLDER, state + '.csv'), index_col='data')


from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_log_error as msle


def create_dataframes(strategy='aggregated', model='linear-regression'):
    id_columns = []
    results = collections.defaultdict(dict)
    
    STATES = ['AC', 'AL', 'AM', 'AP', 'BA', 'CE',
         'DF', 'ES', 'GO', 'MA', 'MT', 'MS',
         'MG', 'PA', 'PB', 'PR', 'PE', 'PI',
         'RJ', 'RN', 'RO', 'RS', 'RR', 'SC',
         'SE', 'SP', 'TO']

    for state in STATES:
        filename = f'result-{state}-{strategy}-{model}.csv'
        results[strategy][model] = pd.read_csv(path.join(RESULTS_FOLDER, 
                                                                  state, 
                                                                  filename), 
                                                        index_col='data')

        id_columns += results[strategy][model].columns.tolist()
    
    id_columns_unique = pd.Series(id_columns).unique()
#     print(id_columns)

    u = pd.DataFrame(columns=['state', 'model', 'metric'] + id_columns_unique.tolist())
    
    return u


def calculate_metrics(y_true, y_hat, metric):
    if metric == 'RMSE':
        m = np.sqrt(mse(y_true, y_hat))
    elif metric == 'MAE':
        m = mae(y_true, y_hat)
    elif metric == 'MSLE':
        m = msle(y_true, y_hat)
    
    return m


def generate_results(strategy='aggregated'):
    results = collections.defaultdict(dict)
    
    df_out = create_dataframes(strategy)

    for state in STATES:
        print(state)
        for model in MODELS:
            filename = f'result-{state}-{strategy}-{model}.csv'
            print(filename)
            results[strategy][model] = pd.read_csv(path.join(RESULTS_FOLDER, 
                                                                      state, 
                                                                      filename), 
                                                            index_col='data')
            for metric in METRICS:
                df_buffer = create_dataframes(strategy)
                for i in results[strategy][model].columns:  
                    # Get dayout
                    dayout = df_state[state]['cases'].index[-1]
                    # Get y_true and y_hat
                    y_hat = results[strategy][model][i].dropna()
                    y_true = df_state[state]['cases']
                    y_true = y_true.loc[y_hat.index.unique()].dropna()
                    y_true, y_hat = utils.check_inputs(y_true, y_hat)

    #                 if state == 'SP':
    #                     print(i)
    #                     print(dayout)
    #                     print(y_true)

                    if y_true.index[-1] == dayout:
                        # print('==========')
                        # print(strategy)
                        # print(model)
                        # print(i)
                        # print(y_hat)
                        # print('==========')
                        break

                    m = calculate_metrics(y_true, y_hat, metric)
                    
                    df_buffer['state'] = [state]
                    df_buffer['model'] = [model]
                    df_buffer['metric'] = [metric]
                    df_buffer[i] = [m]
                  
#                 print(df_buffer)
                df_out = df_out.append(df_buffer, ignore_index = True)
    
    
    return df_out



if __name__ == "__main__":

    df_out = generate_results()
    df_out

    df_out.to_csv('aaa.csv')