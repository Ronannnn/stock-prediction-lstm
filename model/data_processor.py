import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class DataLoader:
    def __init__(self, config):
        df = yf.download(config["stock_code"], start=config["start"], end=config["end"])
        if len(df) == 0:
            raise Exception("No data for stock code" + config["stock_code"])
        split_num = int(len(df) * config["train_test_split"])
        df_with_cols = df.get(config["columns"])
        normalizable = config["normalizable"]
        self.train_data = self.normalize(df_with_cols[:split_num], normalizable)
        self.test_data = self.normalize(df_with_cols[split_num:], normalizable)
        self.train_len = len(self.train_data)
        self.test_len = len(self.test_data)
        self.days_for_predict = config["days_for_predict"]
        self.days_to_predict = config["days_to_predict"]

    def get_windowed_train_data(self):
        return self.get_windowed_data(self.train_data)

    def get_windowed_test_data(self):
        return self.get_windowed_data(self.test_data)

    def get_windowed_data(self, data):
        windowed_data = []
        time_idx = []
        for i in range(len(data) - self.days_for_predict + 1):
            sub_data = data[i:i + self.days_for_predict]
            windowed_data.append(sub_data)
            time_idx.append(data.index.values[i + self.days_for_predict - 1])
        windowed_data = np.array(windowed_data)
        x = windowed_data[:, :-self.days_to_predict]
        y = windowed_data[:, -self.days_to_predict, [1]]
        return x, y, time_idx

    @staticmethod
    def normalize(data, normalizable):
        return pd.DataFrame(
            MinMaxScaler().fit_transform(data),
            columns=data.columns,
            index=data.index
        ) if normalizable else data

    # for linear regression
    def get_linear_train_data(self):
        data, time_idx = self.get_linear_data(self.train_data, True)
        return [[i] for i in range(self.train_len)], data, time_idx

    def get_linear_test_data(self):
        data, time_idx = self.get_linear_data(self.test_data, True)
        return [[i] for i in range(self.train_len, self.train_len + self.test_len, 1)], data, time_idx

    def get_linear_data(self, data, normalizable):
        normalized_data = self.normalize(data, normalizable)
        return normalized_data[:, [1]].ravel(), [[i] for i in data.index.values]
