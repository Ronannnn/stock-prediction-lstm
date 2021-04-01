import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class DataLoader:

    def __init__(self,
                 stock_code,
                 train_test_split_ratio,
                 cols,
                 seq_len,
                 predicted_days,
                 batch_size,
                 normalise,
                 start,
                 end):
        # todo check if already in db
        # todo set proper start and end
        df = yf.download(stock_code, start=start, end=end)
        split_num = int(len(df) * train_test_split_ratio)
        self.train_data = df.get(cols)[:split_num]
        self.test_data = df.get(cols)[split_num:]
        self.train_len = len(self.train_data)
        self.test_len = len(self.test_data)
        self.seq_len = seq_len
        self.predicted_days = predicted_days
        self.batch_size = batch_size
        self.normalize = normalise

    def get_windowed_train_data(self):
        return self.get_windowed_data(self.train_data)

    def get_windowed_test_data(self):
        return self.get_windowed_data(self.test_data)

    def get_predict_data(self):
        windowed_data = []
        time_idx = []
        for i in range(self.test_len - self.seq_len, -1, -self.predicted_days):
            sub_data = self.test_data[i:i + self.seq_len]
            windowed_data.append(sub_data)
            time_idx.append(list(sub_data.index.values[-self.predicted_days:]))
        # normalization
        windowed_data = self.normalize_windows(windowed_data) if self.normalize else np.array(windowed_data)
        # https://www.pythoninformer.com/python-libraries/numpy/index-and-slice/
        x = windowed_data[:, :-self.predicted_days]
        y = windowed_data[:, -self.predicted_days:, [1]]
        return x, y, time_idx

    def get_windowed_data(self, data):
        windowed_data = []
        time_idx = []
        for i in range(len(data) - self.seq_len + 1):
            sub_data = data[i:i + self.seq_len]
            windowed_data.append(sub_data)
            time_idx.append(list(sub_data.index.values[-self.predicted_days:]))
        # normalization
        windowed_data = self.normalize_windows(windowed_data) if self.normalize else np.array(windowed_data)
        # https://www.pythoninformer.com/python-libraries/numpy/index-and-slice/
        x = windowed_data[:, :-self.predicted_days]
        y = windowed_data[:, -self.predicted_days, [0]]
        return x, y, time_idx

    # todo runtime error
    def generate_train_batch(self):
        i = 0
        while i < (self.train_len - self.seq_len + 1):
            x_batch = []
            y_batch = []
            for b in range(self.batch_size):
                if i >= (self.train_len - self.seq_len + 1):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                window = self.train_data[i:i + self.seq_len]
                window = self.normalize_windows(window) if self.normalize else np.array(window)
                x_batch.append(window[:-1])
                y_batch.append(window[-1, [0]])
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    @staticmethod
    def normalize_windows(windowed_data):
        res = []
        scaler = MinMaxScaler()
        for window in windowed_data:
            scaler.fit(window)
            res.append(scaler.transform(window))
        return np.array(res)

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
