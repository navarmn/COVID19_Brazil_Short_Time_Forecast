from datetime import datetime, timedelta
from os import path
import numpy as np
import pandas as pd

import requests, json


DATE_LIST = [
      "03-25-20",
      "03-26-20",
      "03-27-20",
]

LABELS_TO_DROP_OLD = ['cured', 'deaths', 'refuses', 'suspects']
LABELS_TO_DROP = ['internadosDomiciliar', 'internadosEnfermaria',
       'internadosUTI', 'percentual_cured', 'percentual_deaths',
       'percentual_internados', 'percentual_internadosDomiciliar',
       'percentual_internadosEnfermaria', 'percentual_internadosUTI',
       'total_internados_DB']

def get_labels(df):
    df_out = df.loc[df != 0]

    return df_out.values

def count_days(dayone='03-14-2020', date_string='%m-%d-%y'):
    dayone = datetime.strptime(dayone, date_string)
    days_list = np.arange(1, (datetime.today() - dayone).days + 2, 1).tolist()

    return days_list

def days_until_now(dayone='03-14-2020', dayout=datetime.today(), date_string='%m/%d/%y'):
    dayone = datetime.strptime(dayone, date_string)
    days_list = np.arange(1, (dayout - dayone).days + 2, 1).tolist()

    return len(days_list)


def dates_to_future(days=1):
    base = datetime.today()
    date_list = [datetime.strftime(base + timedelta(days=x), '%m-%d-%y') for x in range(days)]

    return date_list


def forecast(model, future=1, dayone='03-14-2020', date_string='%m-%d-%Y', dayout=datetime.today(), 
            date_string_output='%a, %d %b %Y %H:%m:%S'):
    # Count the number of days up now
    days_before = days_until_now(dayone=dayone, dayout=dayout, date_string=date_string)
    # Count the days ahead from now
    days_future = [x for x in range(days_before + 1, future + days_before + 2)]
    # base = datetime.today()
    date_list = [datetime.strftime(dayout + timedelta(days=x), '%m-%d-%y') for x in range(future + 1)]

    # map_iterator = map(lambda x: datetime.strptime(x, date_string).day, days)
    # days_array = np.array(list(map_iterator))

    # predict with the model
    y_hat = model.predict(np.array(days_future).reshape(-1,1))

    out = pd.DataFrame(columns=['ds', 'yhat'])
    out['ds'] = format_date(date_list, date_string_output=date_string_output)
    out['yhat'] = y_hat

    return out

    
def format_date(date_list, date_string_input='%m-%d-%y', date_string_output='%a, %d %b %Y %H:%m:%S'):
    dates = [datetime.strptime(x, date_string_input) for x in date_list]
    dates_tormated = [x.strftime(date_string_output) for x in dates]

    return dates_tormated


# def porra(df):
#     # new_df = df.iloc[-16:].copy()
#     # new_df = df.copy()
#     new_df = df
#     new_df.reset_index(inplace=True)
#     new_df.drop(labels=['cured', 'deaths', 'refuses', 'suspects'], axis=1, inplace=True)
#     new_df.rename(columns={'data': 'ds', 'cases': 'y'}, inplace=True)

#     # new_df['y']=new_df['y']*0.05

#     return new_df


def ajust_df(df, label='cases'):
    df_new = df.loc[df['cases'] != 0]
    # df_new.drop(labels=['cured', 'deaths', 'refuses', 'suspects'], axis=1, inplace=True)
    # df_new.drop(labels=LABELS_TO_DROP + LABELS_TO_DROP_OLD, axis=1, inplace=True)
    df_new = df_new[label].to_frame()
    # df_new = df_new['cases'].reset_index().drop(labels=['index'], axis=1)
    df_new.rename(columns={label: 'y'}, inplace=True)

    # check if there are zeros
    df_new[df_new['y'] == 0] = 1

    # Ajust date
    df_new.index = pd.DatetimeIndex(df_new.index).to_period('D')

    return df_new


def download_state(URL='http://lapisco.fortaleza.ifce.edu.br:3022/api/covid19stats/historyByBrState?State=', state='CE', dump_folder='./.temp', save=False):
    URL = URL + state
    r = requests.get(URL)
    d = json.loads(r.text)
    with open(path.join(dump_folder, '{}.json'.format(state)), 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=4)
    
    df = pd.read_json(path.join(dump_folder, '{}.json'.format(state)))
    df.set_index(df['data'], inplace=True)
    df.drop(columns=['data'], inplace=True)
#     print(df)
    if save:
        df.to_csv(path.join(dump_folder, '{}.csv'.format(state)))
    
    return df


def check_inputs(X, y):
    if X.shape[0] < y.shape[0]:
        diff = np.abs(X.shape[0] - y.shape[0])
        y_new = y[diff:]
        X_new = X
    elif X.shape[0] > y.shape[0]:
        diff = np.abs(X.shape[0] - y.shape[0])
        X_new = X[diff:]
        y_new = y
    else:
        X_new = X
        y_new = y

    return X_new, y_new



class TimesSeriesSplit():
    def __init__(self, X, method='aggregated', days_to_start=7, n_last_days=7):
        self.days_to_start = days_to_start
        self.method = method
        if self.method == 'aggregated':
            self.i = days_to_start  # To fit to array indices
            self.max = X.shape[0] - 1
        else:
            self.i = 0
            self.max = X.shape[0] - 1
        
        self.indices = np.arange(0, self.max, 1)
        self.n_last_days = n_last_days

    def __iter__(self):
        return self

    def __next__(self):
        if self.method == 'aggregated':
            if self.i > self.max:
                raise StopIteration

            # Salve value to be returned
            train_idx = self.indices[:self.i]
            test_idx = self.indices[self.i:]

            # Update i
            self.i += 1

            return train_idx, test_idx

        if self.method == 'windowed':
            if self.i >= self.max:
                raise StopIteration
                
            # Salve value to be returned
            train_idx = self.indices[self.i:self.i + self.n_last_days]
            test_idx = self.indices[self.i + self.n_last_days:]

            # Update i
            self.i += 1

            return train_idx, test_idx



if __name__ == "__main__":
    X = np.arange(1, 15, 1).reshape(-1,1)
    split = TimesSeriesSplit(X, method='windowed')

    for train_idx, test_idx in split:
        print(f'Train set: {X[train_idx]}')
        print(f'Test set: {X[test_idx]}')
        print('==============')



