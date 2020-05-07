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
RESULTS_FOLDER = path.join('results')

# CREATE TEMP FOLDER
TEMPFOLDER = './.temp/'
if not os.path.exists(TEMPFOLDER):
    os.mkdir(TEMPFOLDER)

STATES = ['AC', 'AP', 'AM', 'PA', 'RO', 'RR',
        'TO', 'AL', 'BA', 'CE', 'MA', 'PB',
        'PE', 'PI', 'RN', 'SE', 'DF', 'GO',
        'MT', 'MS', 'ES', 'MG', 'RJ', 'SP', 'PR',
        'SC', 'RS']

# Loda data from state
url = 'http://lapisco.fortaleza.ifce.edu.br:3011/api/covid19stats/listBrStates'
r = requests.get(url)
states = {}

list_dayone = []
list_days = []
list_states = []
list_cases = []
list_deaths = []
list_cured = []

for state in STATES:
    print(state)
    df_state = utils.download_state(state=state)
    # print(df_state.tail())
    
    # Create the output dataframe
    df_filtered = df_state[df_state['cases'] != 0] 
    df_state_yhat = pd.DataFrame(index=df_filtered.index, columns=['yhat'])

    dayone = df_state[df_state['cases'] != 0].index[0]

    days = np.array(utils.count_days(dayone=dayone, date_string='%d/%m/%Y'))

    dayone = datetime.strptime(dayone, '%d/%m/%Y').strftime('%Y/%m/%d')

    list_dayone.append(dayone)
    list_days.append(len(days))
    list_states.append(state)
    list_cases.append(df_filtered['cases'][-1])
    list_cured.append(df_filtered['cured'][-1])
    list_deaths.append(df_filtered['deaths'][-1])


df_out = pd.DataFrame(columns=['state', 'n_days', 'dayone', 'cases', 'cured', 'deaths'])
df_out['state'] = list_states
df_out['n_days'] = list_days
df_out['dayone'] = list_dayone

df_out['cases'] = list_cases
df_out['cured'] = list_cured
df_out['deaths'] = list_deaths

df_out.to_csv(path.join(RESULTS_FOLDER, 'summary_data.csv'), index=None)




