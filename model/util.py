import datetime as dt
import json
import os

from numpy import dtype
from sqlalchemy import NVARCHAR, FLOAT

import matplotlib.pyplot as plt

class Timer:
    """
    for stopwatch
    """

    def __init__(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        res = dt.datetime.now() - self.start_dt
        print('Time taken: %s' % res)
        return res


def convert_dtypes(dtypes):
    """
    For conversions of data types in db
    :param dtypes:
    :return:
    """
    for k, v in dtypes.items():
        if v == dtype('float64'):
            dtypes[k] = FLOAT()
        elif v == dtype('O'):
            dtypes[k] = NVARCHAR(length=255)
    return dtypes


def get_diff_intervals(old_l, old_r, new_l, new_r):
    """
    return intervals that not covered in old one
    If they are not intersected, make it consecutive
    :param old_l:
    :param old_r:
    :param new_l:
    :param new_r:
    :return:
    """
    intervals = []
    if new_l < old_l:
        intervals.append([new_l, old_l - 1])
    if new_r > old_r:
        intervals.append([old_r + 1, new_r])
    return intervals


def load_config():
    config = json.load(open('model/config.json', 'r'))
    if not os.path.exists(config['data']['save_dir']):
        os.makedirs(config['data']['save_dir'])
    return config


def plot_results(predicted_data, true_data, time_idx):
    for i in range(len(predicted_data)):
        plot_result(predicted_data[i], true_data[i], time_idx[i])


def plot_last_results(predicted_data, true_data, time_idx, num):
    for idx in range(len(predicted_data) - num, len(predicted_data)):
        plot_result(predicted_data[idx], true_data[idx], time_idx[idx])


def plot_result(predicted_data, true_data, time_idx):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(time_idx, true_data, label='True Data')
    plt.plot(time_idx, predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()
