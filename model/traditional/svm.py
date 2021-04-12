from model.data_processor import DataLoader
from model.model_abc import Model
from sklearn.svm import SVR

from model.util import load_config, plot, plot_pred_true_result


class SVMModel(Model):

    def __init__(self, kernel, degree, gamma, C):
        self.model = SVR(kernel=kernel, degree=degree, gamma=gamma, C=C)

    def build(self):
        return

    def train(self, X, y, epochs, batch_size):
        self.model.fit(X, y)
        pass

    def train_with_generator(self, data_generator):
        pass

    def predict(self, X):
        return self.model.predict(X)


svm_configs = [
    {
        "kernel": "rbf",
        "degree": 3,
        "C": 1e3,
        "gamma": 'scale'
    },
    # {
    #     "kernel": "linear",
    #     "degree": 3,
    #     "C": 1e3,
    #     "gamma": 'auto'
    # },
    # {
    #     "kernel": "poly",
    #     "degree": 2,
    #     "C": 1e3,
    #     "gamma": 'auto'
    # },
]


def svm_model_test():
    config = load_config()
    data = DataLoader(config["data"])
    for svm_config in svm_configs:
        # get data
        model = SVMModel(
            kernel=svm_config["kernel"],
            degree=svm_config["degree"],
            C=svm_config["C"],
            gamma=svm_config["gamma"]
        )
        x_train, y_train, _ = data.get_linear_train_data()
        x_pred, y_true, time_idx = data.get_linear_test_data()
        # feed in model and get prediction
        y_pred = model.build_train_predict(x_train, y_train, x_pred, -1, -1)
        model.evaluate(y_true, y_pred)
        plot_pred_true_result(time_idx, y_pred, y_true)


if __name__ == '__main__':
    svm_model_test()
