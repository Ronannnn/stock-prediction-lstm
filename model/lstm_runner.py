import model.api as api
from model.data_processor import DataLoader
from model.model_handler import Model
from model.util import plot_result, load_config


def train():
    config = load_config()

    # common var
    seq_len = config['data']['sequence_length']
    batch_size = config['train']['batch_size']

    data = DataLoader(
        stock_code=config['data']['stock_code'],
        train_test_split_ratio=config['data']['train_test_split'],
        cols=config['data']['columns'],
        seq_len=seq_len,
        predicted_days=config['data']['predicted_days'],
        batch_size=batch_size,
        normalise=config['data']['normalize'],
        start=config['data']['start'],
        end=config['data']['end']
    )

    # build models
    for model_config in config['models']:
        if model_config['include'] is False:
            continue
        model = Model()
        model.build(model_config)

        x, y, _ = data.get_windowed_train_data()
        model.train(x, y, config['train']['epochs'], batch_size, config['data']['save_dir'])

        # model.train_generator(
        #     data_generator=data.generate_train_batch(),
        #     epochs=config['train']['epochs'],
        #     batch_size=batch_size,
        #     steps_per_epoch=math.ceil((data.train_len - seq_len) / batch_size),
        #     save_dir=config['data']['save_dir']
        # )

        x_test, y_test, time_idx = data.get_windowed_test_data()
        predictions = model.predict(x_test, batch_size=batch_size)
        plot_result(predictions, y_test, time_idx)
        for i in range(len(predictions)):
            print(
                "['" + str(time_idx[i][0])[0: 10] + "','" + str(y_test[i][0]) + "','" + str(predictions[i][0]) + "'],")

        # x_test, y_test, time_idx = data.get_predict_data()
        # predictions = model.predict(x_test, batch_size=batch_size)
        # plot_last_results(predictions, y_test, time_idx, 5)


if __name__ == '__main__':
    # df = pd.read_csv('data.csv', index_col=0)
    # df = df.get(["Open"])
    # print(df)
    # print(list(df.index.values))
    train()
    # print(api.get_plot_data("AAPL"))
