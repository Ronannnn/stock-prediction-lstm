import os

import numpy as np
import quandl
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# for us dollar index
quandl.ApiConfig.api_key = 'ZmqDKDtks_xNKfdQv-b4'


class DataLoader:
    def __init__(self, config):
        self.stock_code = config["stock_code"]
        self.start = config["start"]
        self.end = config["end"]
        self.train_test_split = config["train_test_split"]
        self.normalizable = config["normalizable"]
        self.days_for_predict = config["days_for_predict"]
        self.days_to_predict = config["days_to_predict"]
        self.train_data, self.test_data = self.fetch_data()
        self.train_len = len(self.train_data)
        self.test_len = len(self.test_data)

    def fetch_data(self):
        stock = self.fetch_yf_stock()
        us_dollar_idx = self.fetch_us_dollar_idx()
        merged_data = stock.merge(us_dollar_idx, left_index=True, right_index=True)
        normalized_data = self.normalize(merged_data, self.normalizable)
        split_num = int(len(normalized_data) * self.train_test_split)
        return normalized_data[:split_num], normalized_data[split_num:]

    def fetch_yf_stock(self):
        data_filename = "model/data/%s_%s_%s.csv" % (self.stock_code, self.start, self.end)
        if not os.path.exists(data_filename):
            stock_data = yf.download(self.stock_code, start=self.start, end=self.end)
            if len(stock_data) == 0:
                raise Exception("No data for stock code %s" % self.stock_code)
            stock_data.to_csv(data_filename)
        else:
            stock_data = pd.read_csv(data_filename, index_col=0)
        return stock_data

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
        else:
            ice_dx = pd.read_csv(data_filename, index_col=0)
        return ice_dx

    def get_columns_num(self):
        return self.test_data.shape[1]

    @staticmethod
    def normalize(data, normalizable):
        return pd.DataFrame(
            MinMaxScaler().fit_transform(data),
            columns=data.columns,
            index=data.index
        ) if normalizable else data

    def get_windowed_train_data(self):
        return self.get_windowed_data(self.train_data)

    def get_windowed_test_data(self):
        return self.get_windowed_data(self.test_data)

    def get_windowed_data(self, data):
        # get col idx of 'Close'
        close_idx = list(data.columns).index('Close')
        val = data.values
        date_idx = data.index.values
        x = []
        y = []
        time_idx = []
        for l in range(len(data) - self.days_for_predict):
            r = l + self.days_for_predict
            x.append(val[l:r])
            y.append([val[r][close_idx]])
            time_idx.append(date_idx[r])
        return np.array(x), np.array(y), time_idx

    # for linear regression
    def get_linear_train_data(self):
        data, time_idx = self.get_linear_data(self.train_data)
        return [[i] for i in range(self.train_len)], data, time_idx

    def get_linear_test_data(self):
        data, time_idx = self.get_linear_data(self.test_data)
        return [[i] for i in range(self.train_len, self.train_len + self.test_len, 1)], data, time_idx

    @staticmethod
    def get_linear_data(data):
        return np.array(data)[:, [1]].ravel(), [[i] for i in data.index.values]
