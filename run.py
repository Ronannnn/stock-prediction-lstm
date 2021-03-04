import json
import os

import math
import matplotlib.pyplot as plt

from data_processor import DataLoader
from model import Model


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
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


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['data']['save_dir']):
        os.makedirs(configs['data']['save_dir'])

    # common var
    seq_len = configs['data']['sequence_length']
    batch_size = configs['train']['batch_size']

    data = DataLoader(
        stock_code=configs['data']['stock_code'],
        train_test_split_ratio=configs['data']['train_test_split'],
        cols=configs['data']['columns'],
        seq_len=seq_len,
        batch_size=batch_size,
        normalise=configs['data']['normalize'],
        start=configs['data']['start'],
        end=configs['data']['end']
    )

    model = Model()
    model.build(configs['models'][0])

    x, y = data.get_windowed_train_data()
    model.train(x, y, configs['train']['epochs'], batch_size, configs['data']['save_dir'])

    # model.train_generator(
    #     data_generator=data.generate_train_batch(),
    #     epochs=configs['train']['epochs'],
    #     batch_size=batch_size,
    #     steps_per_epoch=math.ceil((data.train_len - seq_len) / batch_size),
    #     save_dir=configs['data']['save_dir']
    # )

    x_test, y_test = data.get_windowed_test_data()
    predictions = model.predict(x_test, batch_size=batch_size)

    plot_results(predictions, y_test)


if __name__ == '__main__':
    main()
