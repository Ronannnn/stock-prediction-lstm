from abc import abstractmethod, ABCMeta

from model.util import Timer

timer = Timer()


def wrap_func(func_name, func, **args):
    timer.reset()
    return_val = func(**args)
    timer.stop(msg="[Model] " + func_name)
    return return_val


class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self, x, y):
        pass

    @abstractmethod
    def train_with_generator(self, data_generator):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def build_train_predict(self, x_train, y_train, x_predict):
        wrap_func("Build Stage", self.build)
        wrap_func("Train Stage", self.train, x=x_train, y=y_train)
        predict_data = wrap_func("Predict Stage", self.predict, x=x_predict)
        return predict_data
