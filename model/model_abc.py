from abc import abstractmethod, ABCMeta
from sklearn.metrics import mean_squared_error

from model.util import Timer, plot
import numpy as np

wrap_func_timer = Timer()


def wrap_func(func_name, func, **args):
    wrap_func_timer.reset()
    print("[Model] %s started." % func_name)
    return_val = func(**args)
    wrap_func_timer.stop(msg="[Model] %s finished" % func_name)
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

    def build_train_predict(self, x_train, y_train, x_pred, epochs, batch_size):
        wrap_func("Build Stage", self.build)
        wrap_func("Train Stage", self.train, X=x_train, y=y_train, epochs=epochs, batch_size=batch_size)
        y_pred = wrap_func("Predict Stage", self.predict, X=x_pred)
        return y_pred

    @staticmethod
    def evaluate(y_true, y_pred):
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        print('[Model] Evaluate with RMSE: %s' % rmse)
        return rmse

    def find_best_epoch(self, min_epochs, max_epochs, step, x_train, y_train, x_pred, y_true):
        epoch_loss_time = []
        epoch_timer = Timer()
        for epoch in range(min_epochs, max_epochs + step, step):
            epoch_timer.reset()
            wrap_func("Build Stage", self.build)
            wrap_func("Train Stage, epoch %s" % epoch, self.train, X=x_train, y=y_train, epochs=epoch, batch_size=50)
            y_pred = wrap_func("Predict Stage, epoch %s" % epoch, self.predict, X=x_pred)
            epoch_loss_time.append([
                epoch,
                self.evaluate(y_true, y_pred),
                epoch_timer.stop("[Model] epoch %s finished" % epoch)
            ])
            print()
        # plot
        epoch_loss_time = np.array(epoch_loss_time)
        time_idx = epoch_loss_time[:, [0]].ravel()
        mse = epoch_loss_time[:, [1]].ravel()
        time_taken = [time.total_seconds() for time in epoch_loss_time[:, [2]].ravel()]
        plot(time_idx, {"mse": mse})
        plot(time_idx, {"time taken": time_taken})
        plot(time_idx, {"mse": mse, "time taken": time_taken})
        return epoch_loss_time

    def find_best_batch_size(self, min_batch_size, max_batch_size, step, x_train, y_train, x_pred, y_true):
        batch_size_loss_time = []
        batch_size_timer = Timer()
        for batch_size in range(min_batch_size, max_batch_size + step, step):
            wrap_func("Build Stage", self.build)
            batch_size_timer.reset()
            wrap_func("Train Stage, batch size %s" % batch_size,
                      self.train,
                      X=x_train,
                      y=y_train,
                      epochs=20,
                      batch_size=batch_size)
            y_pred = wrap_func("Predict Stage, batch size %s" % batch_size, self.predict, X=x_pred)
            batch_size_loss_time.append([
                batch_size,
                self.evaluate(y_true, y_pred),
                batch_size_timer.stop("[Model] batch size %s finished" % batch_size)
            ])
            print()
        # plot
        batch_size_loss_time = np.array(batch_size_loss_time)
        time_idx = batch_size_loss_time[:, [0]].ravel()
        mse = batch_size_loss_time[:, [1]].ravel()
        time_taken = [time.total_seconds() for time in batch_size_loss_time[:, [2]].ravel()]
        plot(time_idx, {"mse": mse})
        plot(time_idx, {"time taken": time_taken})
        plot(time_idx, {"mse": mse, "time taken": time_taken})
        return batch_size_loss_time
