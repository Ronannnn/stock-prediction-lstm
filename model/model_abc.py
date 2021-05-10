from abc import abstractmethod, ABCMeta
from sklearn.metrics import mean_squared_error

from model.util import Timer, plot_pred_true_result
import numpy as np

wrap_func_timer = Timer()


def wrap_func(func_name, func, **args):
    wrap_func_timer.reset()
    print("[%s] Processing" % func_name)
    return_val = func(**args)
    wrap_func_timer.stop(msg="[%s]" % func_name)
    return return_val


class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self, X, y, epochs, batch_size):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @staticmethod
    def evaluate(y_true, y_pred):
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))
        print('\tRMSE: %s, R2: %s' % (rmse, r2))
        return round(rmse, 3), round(r2, 3)

    def learn(self, x_train, y_train, x_test, y_test, date_test, y_scaler, epochs, batch_size, plot=True):
        wrap_func("Build", self.build)
        wrap_func("Train", self.train, X=x_train, y=y_train, epochs=epochs, batch_size=batch_size)
        y_pred = wrap_func("Prediction", self.predict, X=x_test)
        y_test = y_scaler.inverse_transform(y_test)
        y_pred = y_scaler.inverse_transform(y_pred)  # this contains one more data than y_test
        rmse, r2 = wrap_func("Evaluation", self.evaluate, y_true=y_test, y_pred=y_pred[:-1])
        if plot:
            plot_pred_true_result(date_test[:-1], y_test, y_pred[:-1])
            print("[Plot] True vs Pred result plotted")
        return rmse, r2
