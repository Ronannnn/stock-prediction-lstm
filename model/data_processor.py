import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class DataLoader:
    def __init__(self,
                 stock_code=None,
                 start=None,
                 end=None,
                 cols=None,
                 train_test_split_ratio=0.75,
                 days_for_predict=50,
                 days_to_predict=1,
                 normalizable=True):
        df = yf.download(stock_code, start=start, end=end)
        if len(df) == 0:
            raise Exception("No data for stock code" + stock_code)
        split_num = int(len(df) * train_test_split_ratio)
        self.train_data = df.get(cols)[:split_num]
        self.test_data = df.get(cols)[split_num:]
        self.train_len = len(self.train_data)
        self.test_len = len(self.test_data)
        self.days_for_predict = days_for_predict
        self.days_to_predict = days_to_predict
        self.normalizable = normalizable

    def get_train_data(self):
        return self.get_data(self.train_data)

    def get_test_data(self):
        return self.get_data(self.test_data)

    def get_data(self, data):
        normalized_data = self.normalize_data(data) if self.normalizable else data
        x = normalized_data[:]
        y = normalized_data[:, [1]]  # close
        time_idx = data.index.values
        return x, y, time_idx

    def get_windowed_train_data(self):
        return self.get_windowed_data(self.train_data)

    def get_windowed_test_data(self):
        return self.get_windowed_data(self.test_data)

    def get_windowed_predict_data(self):
        windowed_data = []
        time_idx = []
        for i in range(self.test_len - self.days_for_predict, -1, -self.days_to_predict):
            sub_data = self.test_data[i:i + self.days_for_predict]
            windowed_data.append(sub_data)
            time_idx.append(list(sub_data.index.values[-self.days_to_predict:]))
        windowed_data = self.normalize_windowed_data(windowed_data) if self.normalizable else np.array(windowed_data)
        x = windowed_data[:, :-self.days_to_predict]
        y = windowed_data[:, -self.days_to_predict:, [1]]
        return x, y, time_idx

    def get_windowed_data(self, data):
        windowed_data = []
        time_idx = []
        for i in range(len(data) - self.days_for_predict + 1):
            sub_data = data[i:i + self.days_for_predict]
            windowed_data.append(sub_data)
            time_idx.append(list(sub_data.index.values[-self.days_to_predict:]))
        windowed_data = self.normalize_windowed_data(windowed_data) if self.normalizable else np.array(windowed_data)
        x = windowed_data[:, :-self.days_to_predict]
        y = windowed_data[:, -self.days_to_predict, [1]]
        return x, y, time_idx

    # todo runtime error
    def generate_train_batch(self, batch_size):
        i = 0
        while i < (self.train_len - self.days_for_predict + 1):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.train_len - self.days_for_predict + 1):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                window = self.train_data[i:i + self.days_for_predict]
                window = self.normalize_windowed_data(window) if self.normalizable else np.array(window)
                x_batch.append(window[:-1])
                y_batch.append(window[-1, [0]])
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    @staticmethod
    def normalize_windowed_data(data):
        res = []
        scaler = MinMaxScaler()
        for window in data:
            scaler.fit(window)
            res.append(scaler.transform(window))
        return np.array(res)

    @staticmethod
    def normalize_data(data):
        scaler = MinMaxScaler()
        scaler.fit(data)
        return scaler.transform(data)

    # db interaction
    def fix_range(self, stock_code, start, end):
        """
        Check if those data of this stock is already in db
        if not, fetch from yfinance api
        """
        return False

    def fetch_from_db(self, stock_codes, start, end):
        return pd.DataFrame

    def fetch_us_dollar_idx(self):
        return 1

    def fetch_interest_rate(self, country):
        return 1
