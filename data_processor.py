import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class DataLoader:

    def __init__(self,
                 stock_code,
                 train_test_split_ratio,
                 cols,
                 seq_len,
                 normalise,
                 start="2019-01-01",
                 end="2020-01-01"):
        # todo check if already in db
        # todo set proper start and end
        df = yf.download(stock_code, start=start, end=end)
        split_num = int(len(df) * train_test_split_ratio)
        self.train_data = df.get(cols).values[:split_num]
        self.test_data = df.get(cols).values[split_num:]
        self.train_len = len(self.train_data)
        self.test_len = len(self.test_data)
        self.seq_len = seq_len
        self.normalize = normalise

    def get_windowed_train_data(self, normalize):
        return self.get_windowed_data(self.train_data, normalize)

    def get_windowed_test_data(self, normalize):
        return self.get_windowed_data(self.test_data, normalize)

    def get_windowed_data(self, data, normalize):
        windowed_data = []
        for i in range(len(data) - self.seq_len + 1):
            windowed_data.append(data[i:i + self.seq_len])
        # normalization
        windowed_data = self.normalize_windows(windowed_data) if normalize else windowed_data
        # https://www.pythoninformer.com/python-libraries/numpy/index-and-slice/
        x = windowed_data[:, :-1]
        y = windowed_data[:, -1, [0]]
        return x, y

    @staticmethod
    def normalize_windows(windowed_data):
        res = []
        scaler = MinMaxScaler()
        for window in windowed_data:
            scaler.fit(window)
            res.append(scaler.transform(window))
        return np.array(res)

    def fetch_us_dollar_idx(self):
        return 1

    def fetch_interest_rate(self, country):
        return 1


if __name__ == '__main__':
    loader1 = DataLoader("GOOGL", 0.5, ["Close", "Volume"], 4, True, "2020-01-01", "2020-01-16")

