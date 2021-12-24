import requests
import json
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import scipy as sc
from scipy import signal

APIKEY='xxxx' # Need to get your own API key from Polygon.io
today=pd.Timestamp('today').strftime("%Y-%m-%d %H:%M:00").split(" ")[0] # Extracting today's date


def fetch_polygon_IO_df_from_URL(URL,data_type):
    columns=['Close','High','Low','Number of Transactions','Open','Time','Volume','Volume Weighted Average']
    results_df=pd.DataFrame(columns=columns)
    try:
        content=requests.get(URL).content
        results_df=pd.DataFrame(json.loads(content)['results'])
        results_df=results_df[['c','h','l','n','o','t','v','vw']]
        results_df.columns=columns
        results_df['Time']=pd.to_datetime(results_df['Time'], unit='ms')
        results_df.set_index("Time", inplace=True)

        if data_type == 'daily':
            results_df.index=results_df.index.strftime("%Y-%m-%d")
            results_df.index.rename("Date", inplace=True)

        if data_type == 'intraday':
            results_df.index= results_df.index.tz_localize("UTC").tz_convert("US/Eastern")
            results_df.index = results_df.index.strftime("%Y-%m-%d %H:%M:00")
    except:
        pass
    return results_df[['Close']]

def preprocess_df(df,data_type):
    df.dropna(axis=1,how='all',inplace=True) # Deleting Empty Columns
    try:
        if data_type =='daily':
            for col in df.columns:
                logged_col = np.log(df[[col]].dropna())  # Logarithm
                detrended_logged_col = sc.signal.detrend(logged_col[col])  # Detrending
                col_norm = np.linalg.norm(detrended_logged_col, 2)
                df[col].loc[logged_col.index] = detrended_logged_col / col_norm  # Normalization

        if data_type =='intraday':
            for col in df.columns:
                col_norm=np.linalg.norm(df[[col]].dropna(),2)
                df[col] = df[col] / col_norm # Normalization

    except:
        pass

    df.fillna(method='ffill', inplace=True)  # Imputing in backwards direction
    df.fillna(method='bfill', inplace=True)  # Imputing in forwards direction
    df=df-df.iloc[0] # Translation by First Value
    return df

class FetchPrices:
    def __init__(self, abbrevs,stock_type):
        self.abbrevs=abbrevs
        self.stock_type=stock_type
        self.daily_URL=''
        self.intraday_URL=''

        if self.stock_type =='stock':
            self.daily_URL='https://api.polygon.io/v2/aggs/ticker/{}/range/1/day/{}/{}?adjusted=true&sort=asc&limit=50000&apiKey={}'
            self.intraday_URL='https://api.polygon.io/v2/aggs/ticker/{}/range/1/minute/{}/{}?adjusted=true&sort=asc&apiKey={}'
        if self.stock_type =='crypto':
            self.daily_URL = 'https://api.polygon.io/v2/aggs/ticker/X:{}USD/range/1/day/{}/{}?adjusted=true&sort=asc&apiKey={}'
            self.intraday_URL = 'https://api.polygon.io/v2/aggs/ticker/X:{}USD/range/1/minute/{}/{}?adjusted=true&sort=asc&apiKey={}'

    def fetch_daily_prices_df(self, start_date='2010-01-01', end_date=today, preprocessed=True, parallelize=True):
        date_range = pd.date_range(start_date,end_date, freq='D').strftime("%Y-%m-%d")
        daily_prices_df = pd.DataFrame(index=date_range)
        daily_prices_df.index.rename("Date", inplace=True)
        daily_URLs = [self.daily_URL.format(abbrev, start_date, end_date, APIKEY) for abbrev in self.abbrevs]
        daily_prices_dfs = None

        if parallelize:
            N=len(self.abbrevs)
            daily_prices_dfs=Parallel(n_jobs=N, verbose=0, prefer='threads')(delayed(fetch_polygon_IO_df_from_URL)(URL,'daily')
                                                                                     for URL in daily_URLs)
        else:
            daily_prices_dfs=[fetch_polygon_IO_df_from_URL(URL,'daily') for URL in daily_URLs]
        daily_prices_df = pd.concat([daily_prices_df]+daily_prices_dfs,axis=1)
        daily_prices_df.columns=self.abbrevs

        if preprocessed:
            daily_prices_df= preprocess_df(daily_prices_df, 'daily')
        return daily_prices_df

    def fetch_intraday_prices_df(self, date=today, preprocessed=True, parallelize=True):
        start_time, end_time = "{} 00:00:00".format(date), "{} 23:59:00".format(date)
        time_range = pd.date_range(start_time, end_time,freq='T').strftime('%Y-%m-%d %H:%M:00')
        intraday_prices_df = pd.DataFrame(index=time_range)
        intraday_prices_df.index.rename("Time", inplace=True)
        intraday_URLs = [self.intraday_URL.format(abbrev, date, date, APIKEY) for abbrev in self.abbrevs]
        intraday_prices_dfs = None

        if parallelize:
            N=len(self.abbrevs)
            intraday_prices_dfs=Parallel(n_jobs=N, verbose=0, prefer='threads')(delayed(fetch_polygon_IO_df_from_URL)(URL,'intraday')
                                                                                     for URL in intraday_URLs)
        else:
            intraday_prices_dfs=[fetch_polygon_IO_df_from_URL(URL,'intraday') for URL in intraday_URLs]
        intraday_prices_df = pd.concat([intraday_prices_df]+intraday_prices_dfs,axis=1)

        trade_start_time, trade_end_time = start_time, end_time

        if self.stock_type=='stock':
            trade_start_time, trade_end_time = "{} 09:30:00".format(date), "{} 16:00:00".format(date)
        if self.stock_type=='crypto':
            trade_start_time, trade_end_time = "{} 00:00:00".format(date), "{} 20:00:00".format(date)

        intraday_prices_df = intraday_prices_df.loc[trade_start_time : trade_end_time] # Restricting to Trading Window
        intraday_prices_df.columns=self.abbrevs
        intraday_prices_df.dropna(axis=1,how='all',inplace=True) # Dropping Empty Columns

        if preprocessed:
            intraday_prices_df = preprocess_df(intraday_prices_df, 'intraday')

        return intraday_prices_df












