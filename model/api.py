import json
import os

from model.data_processor import DataLoader
from model.model_handler import Model


def get_plot_data(stock_code):
    configs = json.load(open('model/config.json', 'r'))
    configs['data']['stock_code'] = stock_code
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
    model = Model()
    model.build(configs['models'][1])

    x, y, _ = data.get_windowed_train_data()
    model.train(x, y, configs['train']['epochs'], batch_size, configs['data']['save_dir'])

    # model.train_generator(
    #     data_generator=data.generate_train_batch(),
    #     epochs=configs['train']['epochs'],
    #     batch_size=batch_size,
    #     steps_per_epoch=math.ceil((data.train_len - seq_len) / batch_size),
    #     save_dir=configs['data']['save_dir']
    # )

    x_test, y_test, time_idx = data.get_windowed_test_data()
    predictions = model.predict(x_test, batch_size=batch_size)
    res = []
    for i in range(len(predictions)):
        res.append([str(time_idx[i][0])[0: 10], str(y_test[i][0]), str(predictions[i][0])])
    return res
