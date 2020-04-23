import pandas as pd 
import os 
from datetime import datetime

import utils
import requests, json

# CREATE TEMP FOLDER
TEMPFOLDER = './.temp/'
if not os.path.exists(TEMPFOLDER):
    os.mkdir(TEMPFOLDER)

# Loda data from state
url = 'http://lapisco.fortaleza.ifce.edu.br:3011/api/covid19stats/listBrStates'
r = requests.get(url)
states = {}

for state in r.json():
    df_state = utils.download_state(state=state['uf'])
    print(df_state.tail())
    # Create the output dataframe
    df_filtered = df_state[df_state['cases'] != 0] 
    # Save dataset
    filename = f'{state["uf"]}.csv'
    df_filtered.to_csv(os.path.join('data', filename))
