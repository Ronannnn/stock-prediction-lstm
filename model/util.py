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
        print(msg + '. Time taken: %s' % res)
        return res


def load_config():
    config = json.load(open('model/config.json', 'r'))
    if not os.path.exists(config['data']['save_dir']):
        os.makedirs(config['data']['save_dir'])
    return config


def plot_result(predicted_data, true_data, time_idx):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(time_idx, true_data, label='True Data')
    plt.plot(time_idx, predicted_data, label='Prediction')
    plt.legend()
    plt.show()
