from CyclicityAnalysis import *
from FetchPrices import *
import os
import pandas as pd
from joblib import Parallel, delayed
import itertools

class LeaderFollowerFrequencyAnalysis:

    def __init__(self,abbrevs, stock_type, start_date='2021-01-01', end_date=today, save_directory=None):
        self.abbrevs=abbrevs
        self.stock_type=stock_type
        self.start_date = start_date
        self.end_date= end_date
        self.save_directory=save_directory


        if self.save_directory != None:
            try:
                os.makedirs(self.save_directory)
            except:
                pass
        self.strong_leader_follower_dates_df=self.get_strong_leader_follower_dates_df(N=10, update=False)
        self.strong_frequency_counts_df=self.get_strong_frequency_counts_df(N=10, update=False)


    def get_topN_intraday_leader_follower_pairs_df(self, date=today, N=10, update=False):
      file_path = "{}/{}.csv".format(self.save_directory, date)
      topN_intraday_leader_follower_pairs_df=None

      if os.path.exists(file_path):
        topN_intraday_leader_follower_pairs_df = pd.read_csv(file_path,index_col=0)

      if os.path.exists(file_path) == False or date == today:
        update=True

      if update:
        try:
          intraday_prices_df=FetchPrices(self.abbrevs, self.stock_type).fetch_intraday_prices_df(date=date,preprocessed=True, parallelize=False)
          intradayCA =CyclicityAnalysis(intraday_prices_df)
          topN_intraday_leader_follower_pairs = intradayCA.get_topN_leader_follower_pairs(N)
          topN_intraday_leader_follower_pairs_df=pd.DataFrame(topN_intraday_leader_follower_pairs, 
          columns=['Leader','Follower'], index=[date]*N)
          topN_intraday_leader_follower_pairs_df.index.rename('Date',inplace=True)
        except:
          topN_intraday_leader_follower_pairs_df=pd.DataFrame(columns=['Leader','Follower'])

        if self.save_directory !=None:
          topN_intraday_leader_follower_pairs_df.to_csv(file_path)

      return topN_intraday_leader_follower_pairs_df

    def get_strong_leader_follower_dates_df(self, N=10, update=False):
      date_range = pd.date_range(self.start_date, self.end_date, freq='D').strftime("%Y-%m-%d")

      topN_intraday_leader_follower_pairs_dfs = Parallel(n_jobs=len(date_range), verbose=0, prefer='threads')\
      (delayed(self.get_topN_intraday_leader_follower_pairs_df)(date,N,update) for date in date_range)

      concatenated_df=pd.concat(topN_intraday_leader_follower_pairs_dfs, axis=0)
      strong_leader_follower_dates_df = concatenated_df.drop_duplicates(keep='first', ignore_index=True)
      strong_dates_lists=[list(concatenated_df[(concatenated_df['Leader'] == pair[0]) & (concatenated_df['Follower'] == pair[-1])].index) \
      for pair in strong_leader_follower_dates_df.values]

      strong_leader_follower_dates_df['Strong Leader Follower Dates']= strong_dates_lists
      strong_leader_follower_dates_df['Number of Strong Leader Follower Dates']=strong_leader_follower_dates_df['Strong Leader Follower Dates'].apply(lambda x: len(x))


      return strong_leader_follower_dates_df

    def get_strong_frequency_counts_df(self,N=10, update=False):
      strong_leader_follower_dates_df = self.get_strong_leader_follower_dates_df(N=N, update=update)

      strong_frequency_counts_df = pd.DataFrame(columns=self.abbrevs, index=self.abbrevs)
      all_pairs=itertools.product(self.abbrevs, repeat= 2)

      for pair in all_pairs:
        try:
          num_strong_leader_follower_dates = strong_leader_follower_dates_df[\
          (strong_leader_follower_dates_df['Leader'] == pair[0]) \
          & (strong_leader_follower_dates_df['Follower'] == pair[-1])]['Number of Strong Leader Follower Dates'].values[-1]
        except:
          num_strong_leader_follower_dates=0

        strong_frequency_counts_df.loc[pair[0],pair[1]] = num_strong_leader_follower_dates
      return strong_frequency_counts_df

    def plot_strong_frequency_counts_df(self, title='Leader Follower Frequency Matrix', color_label='Number of Strong Leader Follower Dates', color_continuous_scale='viridis' ):
      fig=px.imshow(self.strong_frequency_counts_df,
                  title=title,
                  labels=dict(color=color_label),
                  x=self.strong_frequency_counts_df.columns,
                  y=self.strong_frequency_counts_df.columns,
                  color_continuous_scale=color_continuous_scale)
      fig.show()












      
      


      


      

     

    
        