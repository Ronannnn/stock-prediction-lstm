import json
import os

import matplotlib.pyplot as plt

from model.data_processor import DataLoader
from model.model import Model


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


def train():
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
        predicted_days=configs['data']['predicted_days'],
        batch_size=batch_size,
        normalise=configs['data']['normalize'],
        start=configs['data']['start'],
        end=configs['data']['end']
    )

    # build models
    for model_config in configs['models']:
        if model_config['include'] is False:
            continue
        model = Model()
        # model.build(model_config)
        #
        # x, y, _ = data.get_windowed_train_data()
        # model.train(x, y, configs['train']['epochs'], batch_size, configs['data']['save_dir'])
        model.load("saved_models/lstms-20210402014710-e50.h5")

        # model.train_generator(
        #     data_generator=data.generate_train_batch(),
        #     epochs=configs['train']['epochs'],
        #     batch_size=batch_size,
        #     steps_per_epoch=math.ceil((data.train_len - seq_len) / batch_size),
        #     save_dir=configs['data']['save_dir']
        # )

        x_test, y_test, time_idx = data.get_windowed_test_data()
        predictions = model.predict(x_test, batch_size=batch_size)
        plot_result(predictions, y_test, time_idx)
        for i in range(len(predictions)):
            print("['" + str(time_idx[i][0]) + "','" + str(y_test[i][0]) + "','" + str(predictions[i][0]) + "'],")

        # x_test, y_test, time_idx = data.get_predict_data()
        # predictions = model.predict(x_test, batch_size=batch_size)
        # plot_last_results(predictions, y_test, time_idx, 5)


if __name__ == '__main__':
    # df = pd.read_csv('data.csv', index_col=0)
    # df = df.get(["Open"])
    # print(df)
    # print(list(df.index.values))
    train()
