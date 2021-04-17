from model.data_processor import DataLoader
from model.nn import NNModel
from model.util import Timer, load_config

steps = []


def get_plot_data(params):
    timer = Timer()
    config = load_config()
    config['data']['stock_code'] = params['stockCode']
    config['data']['start'] = params['date'][0][0:10]
    config['data']['end'] = params['date'][1][0:10]

    # data loader
    timer.reset()
    data = DataLoader(config["data"])
    steps.append(timer.stop())

    # model builder
    timer.reset()
    model_config = config["models"][1]
    model = NNModel(model_config)
    model.build()
    steps.append(timer.stop())

    # train
    timer.reset()
    x, y, _ = data.get_windowed_train_data()
    model.train(x, y, model_config["epochs"], model_config["batch_size"])
    steps.append(timer.stop())

    # predict
    timer.reset()
    x_pred, y_true, time_idx = data.get_windowed_test_data()
    y_pred = model.predict(x_pred)
    res = []
    for i in range(len(y_pred)):
        res.append([str(time_idx[i])[0: 10], str(y_true[i][0]), str(y_pred[i][0])])
    steps.append(timer.stop())
    rmse = model.evaluate(y_true, y_pred)
    return res, rmse
