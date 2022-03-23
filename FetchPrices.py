import requests
import json
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import scipy as sc
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

APIKEY='xxxx' # You need to obtain your own API key from Polygon.io ! AND DO NOT SHARE YOUR KEY PUBLICLY !
today=pd.Timestamp('today').strftime("%Y-%m-%d") # Extracting today's date


def fetch_POLYGON_IO_df_from_URL(URL, granularity):
    column_rename_dict={'v': 'Volume', 'vw': 'Volume Weighted Average', 'o': 'Open', 'c': 'Close',
                        'h': 'High', 'l': 'Low', 't': 'Time','n': 'Number of Transactions'}
    try:
        contents=json.loads(requests.get(URL).content)
        results_df=pd.DataFrame(contents['results'])
        results_df.rename(columns=column_rename_dict, inplace=True)
        results_df.set_index('Time',inplace=True) 
        results_df.index = pd.to_datetime(results_df.index, unit='ms').tz_localize("UTC").tz_convert('US/Eastern')
        
        if granularity =='daily':
            results_df.index.rename('Date',inplace=True) 
            results_df.index =results_df.index.strftime("%Y-%m-%d")
        if granularity =='intraday':
            results_df.index =results_df.index.strftime("%Y-%m-%d %H:%M:00")
        return results_df[['Close']]
            
    except:
        return pd.DataFrame(columns=['Close'])


class FetchPrices:
    def __init__(self, abbrevs,financial_type):
        self.abbrevs=abbrevs
        self.financial_type=financial_type
        self.daily_URL_template=''
        self.intraday_URL_template=''

        if self.financial_type =='stock':
            self.daily_URL_template = "https://api.polygon.io/v2/aggs/ticker/{}/range/1/day/{}/{}?adjusted=true&sort=asc&apiKey={}"
            self.intraday_URL_template = "https://api.polygon.io/v2/aggs/ticker/{}/range/1/minute/{}/{}?adjusted=true&sort=asc&apiKey={}"
        if self.financial_type =='crypto':
            self.daily_URL_template = 'https://api.polygon.io/v2/aggs/ticker/X:{}USD/range/1/day/{}/{}?adjusted=true&sort=asc&apiKey={}'
            self.intraday_URL_template = 'https://api.polygon.io/v2/aggs/ticker/X:{}USD/range/1/minute/{}/{}?adjusted=true&sort=asc&apiKey={}'
        if self.financial_type == 'forex':
          self.daily_URL_template = "https://api.polygon.io/v2/aggs/ticker/C:{}USD/range/1/day/{}/{}?adjusted=true&sort=asc&apiKey={}"
          self.intraday_URL_template = "https://api.polygon.io/v2/aggs/ticker/C:{}USD/range/1/minute/{}/{}?adjusted=true&sort=asc&apiKey={}"

        

        

    def fetch_daily_prices_df(self,start_date='2010-01-01', end_date=today, preprocessed=True, parallelize=True):
      date_range = pd.date_range(start_date, end_date, freq='D').strftime("%Y-%m-%d")
      daily_prices_df=pd.DataFrame(index=date_range)
      daily_prices_df.index.rename("Date",inplace=True)
      daily_URLs=[self.daily_URL_template.format(abbrev,start_date,end_date,APIKEY) for abbrev in self.abbrevs]
      daily_prices_dfs=[]

      if parallelize:
        daily_prices_dfs=Parallel(n_jobs=len(self.abbrevs),prefer='threads',verbose=0)\
        (delayed(fetch_POLYGON_IO_df_from_URL)(URL,'daily') for URL in daily_URLs)
      else:
        daily_prices_dfs = [fetch_POLYGON_IO_df_from_URL(URL,'daily') for URL in daily_URLs]

      daily_prices_dfs=[df.loc[start_date : end_date] for df in daily_prices_dfs]
      daily_prices_df =pd.concat([daily_prices_df]+ daily_prices_dfs,axis=1)
      daily_prices_df.columns = self.abbrevs

      if preprocessed:
        for abbrev in self.abbrevs:
          try:
            available_data=daily_prices_df[abbrev].dropna()
            logged_data=np.log(available_data) #Log
            detrended_data=sc.signal.detrend(logged_data) #Detrend
            norm=np.linalg.norm(detrended_data,2)
            normalized_data=detrended_data/norm #normalize
            daily_prices_df[abbrev].loc[available_data.index]=normalized_data
          except:
            daily_prices_df.drop(columns=[abbrev],inplace=True)

        daily_prices_df.fillna(method='ffill',inplace=True) # Forward-fill
        daily_prices_df.fillna(method='bfill',inplace=True) # Back-fill
        daily_prices_df=daily_prices_df-daily_prices_df.iloc[0] #Translation by first Value

      return daily_prices_df



    def fetch_intraday_prices_df(self,date=today, preprocessed=True, parallelize=True):
      trade_start_time, trade_end_time ="{} 00:00:00".format(date), "{} 23:59:00".format(date)
      if self.financial_type =='stock':
        trade_start_time, trade_end_time ="{} 09:30:00".format(date), "{} 16:00:00".format(date)
      if self.financial_type =='forex':
        trade_start_time, trade_end_time ="{} 08:00:00".format(date), "{} 17:00:00".format(date)
      

      time_range = pd.date_range(trade_start_time, trade_end_time, freq='T').strftime("%Y-%m-%d %H:%M:00")
      intraday_prices_df=pd.DataFrame(index=time_range)
      intraday_prices_df.index.rename("Date",inplace=True)
      intraday_URLs=[self.intraday_URL_template.format(abbrev, date, date,APIKEY) for abbrev in self.abbrevs]
      intraday_prices_dfs=[]

      if parallelize:
        intraday_prices_dfs=Parallel(n_jobs=len(self.abbrevs),prefer='threads',verbose=0)\
        (delayed(fetch_POLYGON_IO_df_from_URL)(URL,'intraday') for URL in intraday_URLs)
      else:
        intraday_prices_dfs = [fetch_POLYGON_IO_df_from_URL(URL,'intraday') for URL in intraday_URLs]

      intraday_prices_dfs=[df.loc[trade_start_time : trade_end_time] for df in intraday_prices_dfs]
      intraday_prices_df =pd.concat([intraday_prices_df]+ intraday_prices_dfs,axis=1)
      intraday_prices_df.columns = self.abbrevs

      if preprocessed:
        for abbrev in self.abbrevs:
          try:
            available_data=intraday_prices_df[abbrev].dropna()
            norm=np.linalg.norm(available_data,2)
            normalized_data=available_data/norm
            intraday_prices_df[abbrev].loc[available_data.index]=normalized_data
          except:
            intraday_prices_df.drop(columns=[abbrev],inplace=True)

        intraday_prices_df.fillna(method='ffill',inplace=True)
        intraday_prices_df.fillna(method='bfill',inplace=True)
        intraday_prices_df=intraday_prices_df - intraday_prices_df.iloc[0]

      return intraday_prices_df









      





    











