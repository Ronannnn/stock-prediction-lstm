import os
from datetime import datetime, timedelta

import numpy as np
import quandl
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# for us dollar index
quandl.ApiConfig.api_key = 'ZmqDKDtks_xNKfdQv-b4'
datetime_fmt = '%Y-%m-%d'


class DataLoader:
    def __init__(self, config):
        self.stock_code = config["stock_code"]
        self.start = config["start"]
        # since we use the last window to predict future day, so we fetch one more day
        end_date = datetime.strptime(config["end"], datetime_fmt) + timedelta(days=1)
        self.end = end_date.strftime(datetime_fmt)
        self.train_test_split = config["train_test_split"]
        self.normalizable = config["normalizable"]
        self.days_for_predict = config["days_for_predict"]
        self.days_to_predict = config["days_to_predict"]
        self.raw_data, self.data, self.date_idx = self.fetch_data()

    def fetch_data(self):
        stock = self.fetch_yf_stock()
        us_dollar_idx = self.fetch_us_dollar_idx()
        merged_data = stock.merge(us_dollar_idx, left_index=True, right_index=True)
        return merged_data, self.normalize(merged_data, self.normalizable), merged_data.index.values

    def fetch_yf_stock(self):
        data_filename = "model/data/%s_%s_%s.csv" % (self.stock_code, self.start, self.end)
        if not os.path.exists(data_filename):
            stock_data = yf.download(self.stock_code, start=self.start, end=self.end)
            if len(stock_data) == 0:
                raise Exception("No data for stock code %s" % self.stock_code)
            stock_data.to_csv(data_filename)
        return pd.read_csv(data_filename, index_col=0)  # todo why return stock_data from first if is not ok

    def fetch_us_dollar_idx(self):
        """
        https://www.quandl.com/data/CHRIS/ICE_DX1-US-Dollar-Index-Futures-Continuous-Contract-1-DX1-Front-Month
        :return:
        """
        data_filename = "model/data/%s_%s_%s.csv" % ('ice-dx1', self.start, self.end)
        if not os.path.exists(data_filename):
            ice_dx = quandl.get("CHRIS/ICE_DX1", start_date=self.start, end_date=self.end)
            ice_dx = ice_dx.get(["Settle"])
            if len(ice_dx) == 0:
                raise Exception("No data for stock code %s" % self.stock_code)
            ice_dx.to_csv(data_filename)
        return pd.read_csv(data_filename, index_col=0)

    def get_columns_num(self):
        return self.data.shape[1]

    @staticmethod
    def normalize(data, normalizable):
        return pd.DataFrame(
            MinMaxScaler().fit_transform(data),
            columns=data.columns,
            index=data.index
        ) if normalizable else data

    def get_windowed_data(self):
        windowed_data = []
        windowed_date_idx = []
        for idx in range(len(self.data) - self.days_for_predict):
            windowed_data.append(self.data[idx: idx + self.days_for_predict + 1])
            windowed_date_idx.append(self.date_idx[idx: idx + self.days_for_predict + 1])
        windowed_data = np.array(windowed_data)
        windowed_date_idx = np.array(windowed_date_idx)
        split_num = int(windowed_data.shape[0] * self.train_test_split)
        close_idx = list(self.data.columns).index('Close')
        # train data
        x_train = windowed_data[:split_num, :-1]
        y_train = windowed_data[:split_num, -1, [close_idx]]
        date_train = windowed_date_idx[:split_num, -1]
        # test data
        x_test = windowed_data[split_num:, :-1]  # use the last window to predict next future day
        y_test = windowed_data[split_num:-1, -1, [close_idx]]
        date_test = windowed_date_idx[split_num:, -1]
        return x_train, y_train, date_train, x_test, y_test, date_test

    def get_min_max_scaler(self):
        close_idx = list(self.data.columns).index('Close')
        # min max scaler for inversion of predict data
        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(np.array(self.raw_data)[:, [close_idx]])
        return min_max_scaler
