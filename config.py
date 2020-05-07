from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from statsmodels.tsa.api import Holt, ARMA

from sklearn.base import BaseEstimator

from datetime import datetime

from utils import *



MODELS = {
    'linear-regression': LinearRegression(),
    'gaussian-process': GaussianProcessRegressor(),
    'mlp': MLPRegressor(),
    'svr': SVR(),

}

MODELS_EXPONENTIAL = {
    # 'arma': 10,
    'holt': 10,
}

DAYS_TO_PREDICT = 14
DAYS_TO_START_TRAINING = 7

STRATEGIES = [
    'aggregated',
    'windowed'
]



class HoltLearner(BaseEstimator):
    def __init__(self, dayone, dayout, smoothing_level=0.5, smoothing_slope=0.05, date_string='%m-%d-%Y', date_string_output='%a, %d %b %Y %H:%m:%S'):
        self.smoothing_level = smoothing_level
        self.smoothing_slope = smoothing_slope
        self.dayone = dayone
        self.dayout = dayout
        self.date_string = date_string
        self.date_string_output = date_string_output

    def fit(self, df):
        self.df = ajust_df(df)
        self.model = Holt(self.df['y'], exponential=True).fit(smoothing_level=0.5, smoothing_slope=0.05, optimized=False) 

        return self

    def predict(self, forecast=1):
        series_out = self.model.forecast(forecast)
        # Format date:
        # Count the number of days up now
        days_before = days_until_now(dayone=self.dayone, dayout=self.dayout, date_string=self.date_string)
        # Count the days ahead from now
        days_future = [x for x in range(days_before + 1, forecast + days_before + 2)]
        # base = datetime.today()
        date_list = [datetime.strftime(self.dayout + timedelta(days=x), '%m-%d-%y') for x in range(forecast + 1)]

        # df_out = pd.DataFrame(index=series_out.index.strftime('%d/%m/%Y'), columns=['yhat'],
        #                         data=series_out.values)
        # df_out.index.name = 'ds'

        df_out = pd.DataFrame(columns=['yhat'])
        df_out['ds'] = format_date(date_list[1:], date_string_output=self.date_string_output)
        df_out['yhat'] = series_out.values

        return df_out


class ARMALearner(HoltLearner):
    def __init__(self, dayone, dayout, order=(1,0), smoothing_slope=0.05, date_string='%m-%d-%Y', date_string_output='%a, %d %b %Y %H:%m:%S'):
        self.dayone = dayone
        self.dayout = dayout
        self.date_string = date_string
        self.date_string_output = date_string_output
        self.order = order


    def fit(self, df):
        self.df = ajust_df(df)
        self.model = ARMA(self.df['y'], self.order).fit() 

        return self
        
        
        
        