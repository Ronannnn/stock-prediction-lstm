from abc import abstractmethod, ABCMeta
from sklearn.metrics import mean_squared_error

from model.util import Timer

timer = Timer()


def wrap_func(func_name, func, **args):
    timer.reset()
    print("[Model] %s started." % func_name)
    return_val = func(**args)
    timer.stop(msg="[Model] %s finished" % func_name)
    return return_val


class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def train_with_generator(self, data_generator):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def build_train_predict(self, x_train, y_train, x_predict):
        wrap_func("Build Stage", self.build)
        wrap_func("Train Stage", self.train, X=x_train, y=y_train)
        predict_data = wrap_func("Predict Stage", self.predict, X=x_predict)
        return predict_data

    @staticmethod
    def evaluate(y_true, y_pred):
        print('[Model] Evaluate with MSE: %s' % mean_squared_error(y_true, y_pred))
