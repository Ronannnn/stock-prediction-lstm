import json
import os

from model.data_processor import DataLoader
from model.model_handler import Model
from model.util import Timer

steps = []


def get_plot_data(stock_code):
    configs = json.load(open('model/config.json', 'r'))
    configs['data']['stock_code'] = stock_code
    if not os.path.exists(configs['data']['save_dir']):
        os.makedirs(configs['data']['save_dir'])

    # data loader
    timer = Timer()
    data = DataLoader(
        stock_code=configs['data']['stock_code'],
        train_test_split_ratio=configs['data']['train_test_split'],
        cols=configs['data']['columns'],
        seq_len=configs['data']['sequence_length'],
        predicted_days=configs['data']['predicted_days'],
        batch_size=configs['train']['batch_size'],
        normalise=configs['data']['normalize'],
        start=configs['data']['start'],
        end=configs['data']['end']
    )
    steps.append(timer.stop())

    # model builder
    timer = Timer()
    model = Model()
    model.build(configs['models'][1])
    steps.append(timer.stop())

    # train
    timer = Timer()
    x, y, _ = data.get_windowed_train_data()
    model.train(x, y, configs['train']['epochs'], configs['train']['batch_size'], configs['data']['save_dir'])
    steps.append(timer.stop())

    # predict
    timer = Timer()
    x_test, y_test, time_idx = data.get_windowed_test_data()
    predictions = model.predict(x_test, batch_size=configs['train']['batch_size'])
    res = []
    for i in range(len(predictions)):
        res.append([str(time_idx[i][0])[0: 10], str(y_test[i][0]), str(predictions[i][0])])
    steps.append(timer.stop())
    return res
