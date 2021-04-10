from model.data_processor import DataLoader
from model.nn import NNModel
from model.util import Timer, load_config

steps = []


def get_plot_data(stock_code):
    timer = Timer()
    config = load_config()
    config['data']['stock_code'] = stock_code

    # data loader
    timer.reset()
    data = DataLoader(config["data"])
    steps.append(timer.stop())

    # model builder
    timer.reset()
    model = NNModel(config["models"][1])
    model.build()
    steps.append(timer.stop())

    # train
    timer.reset()
    x, y, _ = data.get_windowed_train_data()
    model.train(x, y)
    steps.append(timer.stop())

    # predict
    timer.reset()
    x_test, y_test, time_idx = data.get_windowed_test_data()
    predictions = model.predict(x_test)
    res = []
    for i in range(len(predictions)):
        res.append([str(time_idx[i][0])[0: 10], str(y_test[i][0]), str(predictions[i][0])])
    steps.append(timer.stop())
    return res
