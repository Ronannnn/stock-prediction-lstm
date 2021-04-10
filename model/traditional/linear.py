from sklearn import linear_model

from model.data_processor import DataLoader
from model.model_abc import Model
from model.util import load_config, plot_result


class LinearModel(Model):

    def build(self):
        self.model = linear_model.LinearRegression()

    def train(self, x, y):
        self.model.fit(x, y)

    def train_with_generator(self, data_generator):
        pass

    def predict(self, x):
        return self.model.predict(x)


def linear_model_test():
    config = load_config()
    data = DataLoader(config["data"])
    model = LinearModel()
    # get data
    close_data_train, _ = data.get_linear_train_data()
    close_data_test, time_idx = data.get_linear_test_data()
    # feed in model and get prediction
    x_train = [[i] for i in range(len(close_data_train))]
    y_train = close_data_train
    x_pred = [[i] for i in range(len(close_data_train), len(close_data_train) + len(close_data_test), 1)]
    y_pred = model.build_train_predict(x_train, y_train, x_pred)
    model.evaluate(close_data_test, y_pred)
    plot_result(y_pred, close_data_test, time_idx)


if __name__ == '__main__':
    linear_model_test()
