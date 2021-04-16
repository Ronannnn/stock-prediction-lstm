from sklearn import linear_model

from model.data_processor import DataLoader
from model.model_abc import Model
from model.util import load_config, plot, plot_pred_true_result


class LinearModel(Model):

    def build(self):
        self.model = linear_model.LinearRegression()

    def train(self, X, y, epochs, batch_size):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


def linear_model_test():
    config = load_config()
    data = DataLoader(config["data"])
    model = LinearModel()
    # get data
    x_train, y_train, _ = data.get_linear_train_data()
    x_pred, y_true, time_idx = data.get_linear_test_data()
    # feed in model and get prediction
    y_pred = model.build_train_predict(x_train, y_train, x_pred, -1, -1)
    model.evaluate(y_true, y_pred)
    plot_pred_true_result(time_idx, y_pred, y_true)


if __name__ == '__main__':
    linear_model_test()
