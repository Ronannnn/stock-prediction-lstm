import model.api as api
from model.data_processor import DataLoader
from model.windowed_trainer import Model
from model.util import plot_result, load_config


def train():
    config = load_config()

    data = DataLoader(
        stock_code=config['data']['stock_code'],
        train_test_split_ratio=config['data']['train_test_split'],
        cols=config['data']['columns'],
        days_for_predict=config['data']['days_for_predict'],
        days_to_predict=config['data']['days_to_predict'],
        normalizable=config['data']['normalizable'],
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
        model.train(x, y, config['data']['epochs'], config['data']['batch_size'], config['data']['save_dir'])

        # model.train_generator(
        #     data_generator=data.generate_train_batch(),
        #     epochs=config['data']['epochs'],
        #     batch_size=batch_size,
        #     steps_per_epoch=math.ceil((data.train_len - seq_len) / batch_size),
        #     save_dir=config['data']['save_dir']
        # )

        x_test, y_test, time_idx = data.get_windowed_test_data()
        predictions = model.predict(x_test, batch_size=config['data']['batch_size'])
        plot_result(predictions, y_test, time_idx)

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
