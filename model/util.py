import datetime as dt
import json
import os

import matplotlib.pyplot as plt


class Timer:
    """
    for stopwatch
    """

    def reset(self):
        self.start_dt = dt.datetime.now()

    def stop(self, msg=""):
        res = dt.datetime.now() - self.start_dt
        print(msg + ' Time taken: %s' % res)
        return res


def load_config():
    config = json.load(open('model/config.json', 'r'))
    if not os.path.exists(config['data']['save_dir']):
        os.makedirs(config['data']['save_dir'])
    return config


def plot(time_idx, data):
    """
    data format:
    {
        data description: data
    }
    """
    for key in data:
        plt.plot(time_idx, data[key], label=key)
    plt.xticks(time_idx[::5], rotation='vertical')
    plt.legend()
    plt.show()


def plot_pred_true_result(time_idx, y_true, y_pred):
    plot(time_idx, {
        "true data": y_true,
        "1-day pred": y_pred,
    })
