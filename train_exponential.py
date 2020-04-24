import joblib
import pandas as pd 
import numpy as np
import os 
from datetime import datetime

import utils
import requests, json
import time

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
    error = 0
    for model_name, _ in MODELS_EXPONENTIAL.items():
        for strategy in STRATEGIES:
            print('Training model {} for state: {}'.format(model_name, state['uf']))
            df_state = utils.download_state(state=state['uf'])
            print(df_state.tail())
            
            # Create the output dataframe
            df_filtered = df_state[df_state['cases'] != 0] 
            #
            df_state_yhat = pd.DataFrame(index=df_filtered.index, columns=['yhat'])

            dayone = df_state[df_state['cases'] != 0].index[0]
            days = np.array(utils.count_days(dayone=dayone, date_string='%d/%m/%Y'))
            X = days.reshape(-1,1)
            y = utils.get_labels(df_state['cases']).reshape(-1,1)

            tcvs = utils.TimesSeriesSplit(X, method=strategy)

            for train_idx, _ in tcvs:
                X_train, y_train = X[train_idx], y[train_idx]
                # Split train and test set
                df_train = df_filtered.iloc[train_idx]

                if df_train.shape[0] == 1:
                    continue

                dayout = df_filtered.iloc[train_idx + 1].index[-1]   
                print('{} - dayone: {}'.format(state, dayone))
                print('{} - dayout: {}'.format(state, dayout))
                dayout = datetime.strptime(dayout, '%d/%m/%Y')

                print('Print inputs/outputs shapes: \n X: {} \n y: {}' .format(X_train.shape, y_train.shape))
                X_train, y_train = utils.check_inputs(X_train, y_train)
                print('Print inputs/outputs shapes corrected: \n X: {} \n y: {}' .format(X_train.shape, y_train.shape))

                # try:
                #     model = HoltLearner(dayone=dayone, dayout=dayout, date_string_output='%d/%m/%Y')
                #     model.fit(df_train)
                #     df_out = model.predict(forecast=DAYS_TO_PREDICT)
                # except:
                #     error += 1
                #     continue

                try:
                    model = HoltLearner(dayone=dayone, dayout=dayout, date_string='%d/%m/%Y', date_string_output='%d/%m/%Y')
                    model.fit(df_train)
                    df_out = model.predict(forecast=DAYS_TO_PREDICT)
                    # Check if predictions are belows yesterday
                    last_value = df_filtered.cases.iloc[y_train.shape[0]]
                    df_out['yhat'] = utils.rescale_yhat(df_out['yhat'].values, last_value)

                except:
                    error += 1
                    continue
                
                print(df_out)
                df_out.set_index('ds', inplace=True)
                # Set vars
                column_name = f'yhat_model_{int(X_train[0])}_to_{int(X_train[-1])}'
                # Update columns
                new_column = pd.Series(data=df_out.yhat, name=column_name, index=df_out.index)
                df_state_yhat[column_name] = np.nan
                df_state_yhat.update(new_column)

            # Drop dataframe
            state_uf = state['uf']
            folder = os.path.join('results', f'{state_uf}')
            if not os.path.exists(folder):
                os.mkdir(folder)

            filename = f'result-{state_uf}-{strategy}-exponential-{model_name}.csv'
            df_state_yhat.drop(columns=['yhat'], inplace=True)
            df_state_yhat.to_csv(os.path.join(folder, filename))

        print(f'#Error: {error}')
        # time.sleep(0.5)






