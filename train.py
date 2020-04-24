import joblib
import pandas as pd 
import numpy as np
import os 
from datetime import datetime

import utils
import requests, json

from config import *

# Global variables
MODELS_FOLDER = os.path.join('..', 'models')
CEARA_DATA = 'ceara.csv'
TIMESTAMP = datetime.strftime(datetime.now(), '%y-%m-%d')


from decouple import config as cfg
# DATA_FOLDER_PROCESSED = cfg('DATA_FOLDER_PROCESSED', cast=str)
DAYS_TO_TRAIN = cfg('DAYS_TO_TRAIN', default=10, cast=int)

# CREATE TEMP FOLDER
TEMPFOLDER = './.temp/'
if not os.path.exists(TEMPFOLDER):
    os.mkdir(TEMPFOLDER)

# Loda data from state
url = 'http://lapisco.fortaleza.ifce.edu.br:3011/api/covid19stats/listBrStates'
r = requests.get(url)
states = {}

for state in r.json():
    for model_name, model in MODELS.items():
        for strategy in STRATEGIES:
            print('Training model {} for state: {}'.format(model_name, state['uf']))
            df_state = utils.download_state(state=state['uf'])
            print(df_state.tail())
            
            # Create the output dataframe
            df_filtered = df_state[df_state['cases'] != 0] 
            df_state_yhat = pd.DataFrame(index=df_filtered.index, columns=['yhat'])

            dayone = df_state[df_state['cases'] != 0].index[0]
            days = np.array(utils.count_days(dayone=dayone, date_string='%d/%m/%Y'))
            X = days.reshape(-1,1)
            y = utils.get_labels(df_state['cases']).reshape(-1,1)

            tcvs = utils.TimesSeriesSplit(X, method=strategy)

            for train_idx, _ in tcvs:
                # Split train and test set
                X_train, y_train = X[train_idx], y[train_idx]

                dayout = df_filtered.iloc[train_idx + 1].index[-1]   
                print('{} - dayone: {}'.format(state, dayone))
                print('{} - dayout: {}'.format(state, dayout))
                dayout = datetime.strptime(dayout, '%d/%m/%Y')

                print('Print inputs/outputs shapes: \n X: {} \n y: {}' .format(X_train.shape, y_train.shape))
                X_train, y_train = utils.check_inputs(X_train, y_train)
                print('Print inputs/outputs shapes corrected: \n X: {} \n y: {}' .format(X_train.shape, y_train.shape))
                
                # Fit model
                model.fit(X_train, y_train)
                # Forecast
                df_out = utils.forecast(model, future=DAYS_TO_PREDICT, dayone=dayone, date_string='%d/%m/%Y', 
                                        dayout=dayout, date_string_output='%d/%m/%Y')

                # Check if predictions are belows yesterday
                last_value = df_filtered.cases.iloc[y_train.shape[0]]
                df_out['yhat'] = utils.rescale_yhat(df_out['yhat'].values, last_value)

                print(df_out)
                df_out.set_index('ds', inplace=True)
                # Set vars
                column_name = f'yhat_model_{int(X_train[0])}_to_{int(X_train[-1])}'
                # Update columns
                new_column = pd.Series(data=df_out.yhat, name=column_name, index=df_out.index)
                df_state_yhat[column_name] = np.nan
                df_state_yhat.update(new_column)

            # Save dataframe
            state_uf = state['uf']
            folder = os.path.join('results', f'{state_uf}')
            if not os.path.exists(folder):
                os.mkdir(folder)    
            
            df_state_yhat.drop(['yhat'], axis=1, inplace=True)
            filename = f'result-{state_uf}-{strategy}-{model_name}.csv'
            df_state_yhat.to_csv(os.path.join(folder, filename))





