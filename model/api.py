from model.data_processor import DataLoader
from model.nn import NNModel
from model.util import Timer, load_config
import numpy as np

steps = []


def get_plot_data(params):
    timer = Timer()
    config = load_config()
    data_config = config['data']
    data_config['stock_code'] = params['stockCode']
    data_config['start'] = params['date'][0][0:10]
    data_config['end'] = params['date'][1][0:10]

    # data loader
    timer.reset()
    data = DataLoader(config["data"])
    steps.append(timer.stop())

    # model builder
    timer.reset()
    model_config = config["models"][0]
    model = NNModel(data_config, model_config, data.get_columns_num())
    model.build()
    steps.append(timer.stop())

    x_train, y_train, date_train, x_test, y_test, date_test, k_line_data = data.get_windowed_data()

    # train
    timer.reset()
    model.train(x_train, y_train, model_config["epochs"], model_config["batch_size"])
    steps.append(timer.stop())

    # predict
    timer.reset()
    y_pred = model.predict(x_test)
    min_max_scaler = data.get_min_max_scaler()
    y_test = min_max_scaler.inverse_transform(y_test)
    y_test = np.append(y_test, [[-1]], axis=0)  # add empty value so it has the same size as y_pred and date
    y_pred = min_max_scaler.inverse_transform(y_pred)
    res = []
    for i in range(len(y_pred)):
        res.append([
            str(date_test[i])[0: 10],  # date
            y_test[i][0],  # y_test
            y_pred[i][0],  # y_pred
            k_line_data.iloc[i, 0],  # Open
            k_line_data.iloc[i, 1],  # Close
            k_line_data.iloc[i, 2],  # Low
            k_line_data.iloc[i, 3],  # High
        ])
    steps.append(timer.stop())
    rmse, r2 = model.evaluate(y_test, y_pred)
    return res, rmse, r2
