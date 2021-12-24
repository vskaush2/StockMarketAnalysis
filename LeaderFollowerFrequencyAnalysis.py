from CyclicityAnalysis import *
from FetchPrices import *
import os

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

        self.strong_leader_follower_relationship_days_df= self.get_strong_leader_follower_relationship_days_df(N=10,update=False)
        self.strong_frequency_count_df=self.get_strong_frequency_count_df()

    def get_topN_intraday_leader_follower_df(self, date, N=10, update=False):
        topN_intraday_leader_follower_df = pd.DataFrame(columns=['Leader', 'Follower'])
        file_path = "{}/{}.csv".format(self.save_directory, date)

        if os.path.exists(file_path):
            topN_intraday_leader_follower_df = pd.read_csv(file_path, index_col=0)
        else:
            update = True

        if update:
            try:
                intraday_prices_df = FetchPrices(self.abbrevs,self.stock_type).fetch_intraday_prices_df(date)
                CA = CyclicityAnalysis(intraday_prices_df)
                top_pairs = CA.get_topN_leader_follower_pairs(N)
                topN_intraday_leader_follower_df = pd.DataFrame(top_pairs, columns=['Leader', 'Follower'])
                topN_intraday_leader_follower_df['Date'] = date
                topN_intraday_leader_follower_df.set_index("Date", inplace=True)
                print("SUCCESS OBTAINING TOP {} INTRADAY LEADER FOLLOWERS FOR {} ...".format(N, date))
            except:
                print("FAILED OBTAINING TOP {} INTRADAY LEADER FOLLOWERS FOR {} ...".format(N, date))

            if self.save_directory != None:
                topN_intraday_leader_follower_df.to_csv(file_path)

        return topN_intraday_leader_follower_df

    def get_strong_leader_follower_relationship_days_df(self,N=10,update=False):
        date_range = pd.date_range(self.start_date, self.end_date, freq='D').strftime('%Y-%m-%d')
        topN_intraday_leader_follower_dfs = [self.get_topN_intraday_leader_follower_df(date, N, update) for date in date_range]

        concatenated_topN_intraday_leader_follower_df = pd.concat(topN_intraday_leader_follower_dfs, axis=0)
        strong_leader_follower_relationship_days_df = concatenated_topN_intraday_leader_follower_df.value_counts().reset_index()
        strong_leader_follower_relationship_days_df.rename(columns={0: 'Number of Strong Leader Follower Dates'},inplace=True)

        leader_follower_pairs = strong_leader_follower_relationship_days_df[['Leader', 'Follower']].values

        strong_leader_follower_dates = [concatenated_topN_intraday_leader_follower_df[(concatenated_topN_intraday_leader_follower_df['Leader'] == pair[0]) \
                                            & (concatenated_topN_intraday_leader_follower_df['Follower'] == pair[-1])].index
                                        for pair in leader_follower_pairs]

        strong_leader_follower_relationship_days_df['Strong Leader Follower Dates'] = [list(x) for x in strong_leader_follower_dates]

        return strong_leader_follower_relationship_days_df

    def get_strong_frequency_count_df(self):

        frequency_count_df = pd.DataFrame(columns= self.abbrevs, index= self.abbrevs)

        for i in range(len(self.strong_leader_follower_relationship_days_df)):
            leader, follower, frequency_count = self.strong_leader_follower_relationship_days_df.iloc[i]['Leader'], \
                                                self.strong_leader_follower_relationship_days_df.iloc[i]['Follower'], \
                                                self.strong_leader_follower_relationship_days_df.iloc[i]['Number of Strong Leader Follower Dates']

            frequency_count_df.loc[leader, follower] = frequency_count # Filling nonzero counts first.

        frequency_count_df.fillna(0, inplace=True) # All other counts are 0.
        return frequency_count_df

    def plot_strong_frequency_count_df(self, title='Strong Leader Follower Days Frequency Count Matrix',
                                       color_label='Number of Strong Leader Follower Days',
                                       color_continuous_scale='Bluered'):
        fig = px.imshow(self.strong_frequency_count_df,
                        title=title,
                        labels=dict(color=color_label),
                        x=self.strong_frequency_count_df.columns,
                        y=self.strong_frequency_count_df.columns,
                        color_continuous_scale=color_continuous_scale)
        fig.update_xaxes(side='top')
        fig.show()


