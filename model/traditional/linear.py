from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential

from model.data_processor import DataLoader
from model.util import load_config, plot_result


def train():
    configs = load_config()

    data = DataLoader(
        stock_code=configs['data']['stock_code'],
        train_test_split_ratio=configs['data']['train_test_split'],
        cols=configs['data']['columns'],
        normalise=configs['data']['normalize'],
        start=configs['data']['start'],
        end=configs['data']['end']
    )

    # build model
    model = Sequential()
    model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mse', optimizer='adam')

    # train
    x_train, y_train, _ = data.get_train_data()
    callbacks = [
        EarlyStopping(monitor='loss', patience=2),  # Stop after 2 epochs whose loss is no longer decreasing
    ]
    model.fit(x_train, y_train, epochs=configs['train']['epochs'], batch_size=configs['train']['batch_size'], callbacks=callbacks, verbose=2)
    x_test, y_test, time_idx = data.get_test_data()
    predict = model.predict(x_test)
    plot_result(predict, y_test, time_idx)


if __name__ == '__main__':
    train()
