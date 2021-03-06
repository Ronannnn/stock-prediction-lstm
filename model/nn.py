import datetime as dt
import os

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, Bidirectional
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.metrics import MeanAbsolutePercentageError, RootMeanSquaredError

from model.data_processor import DataLoader
from model.model_abc import Model

from model.util import load_config


def new_dense(layer_config):
    neuron_num = layer_config['neuron_num'] if 'neuron_num' in layer_config else None
    input_dim = layer_config['feature_num'] if 'feature_num' in layer_config else None
    kernel_initializer = layer_config['kernel_initializer'] if 'kernel_initializer' in layer_config else None
    activation = layer_config['activation'] if 'activation' in layer_config else None
    return Dense(neuron_num, input_dim=input_dim, kernel_initializer=kernel_initializer, activation=activation)


def new_lstm(layer_config):
    neuron_num, activation, return_seq, days_for_predict, feature_num = lstm_config(layer_config)
    if days_for_predict is None or feature_num is None:
        return LSTM(neuron_num, activation=activation, return_sequences=return_seq)
    else:
        return LSTM(
            neuron_num,
            activation=activation,
            input_shape=(days_for_predict, feature_num),
            return_sequences=return_seq,
        )


def new_bi_lstm(layer_config):
    neuron_num, activation, return_seq, days_for_predict, feature_num = lstm_config(layer_config)
    if days_for_predict is None or feature_num is None:
        return Bidirectional(LSTM(neuron_num, activation=activation, return_sequences=return_seq))
    else:
        return Bidirectional(
            LSTM(neuron_num, activation=activation, return_sequences=return_seq),
            input_shape=(days_for_predict, feature_num)
        )


def lstm_config(layer_config):
    neuron_num = layer_config['neuron_num'] if 'neuron_num' in layer_config else None
    activation = layer_config['activation'] if 'activation' in layer_config else 'tanh'
    return_seq = layer_config['return_seq'] if 'return_seq' in layer_config else False
    days_for_predict = layer_config['days_for_predict'] if 'days_for_predict' in layer_config else None
    feature_num = layer_config['feature_num'] if 'feature_num' in layer_config else None
    return neuron_num, activation, return_seq, days_for_predict, feature_num


def new_dropout(layer_config):
    dropout_rate = layer_config['rate'] if 'rate' in layer_config else None
    return Dropout(dropout_rate)


def new_repeat_vector(layer_config):
    neuron_num = layer_config['neuron_num'] if 'neuron_num' in layer_config else None
    return RepeatVector(neuron_num)


def new_time_distributed_dense(layer_config):
    neuron_num = layer_config['neuron_num'] if 'neuron_num' in layer_config else None
    return TimeDistributed(Dense(neuron_num))


layer_dict = {
    'dense': new_dense,
    'lstm': new_lstm,
    'bilstm': new_bi_lstm,
    'dropout': new_dropout,
    'repeat': new_repeat_vector,
    'time_dense': new_time_distributed_dense
}


class NNModel(Model):

    def __init__(self, data_config, model_config, feature_num):
        self.model_config = model_config
        self.days_for_predict = data_config['days_for_predict']
        self.days_to_predict = data_config['days_to_predict']
        self.feature_num = feature_num
        self.filename = os.path.join(
            data_config['save_dir'],
            '%s-%s.h5' % (self.model_config['name'], dt.datetime.now().strftime('%Y%m%d%H%M%S')))
        self.verbose = data_config['verbose']

    def load(self, file):
        self.model = load_model(file)

    def build(self, print_model=False):
        self.model = Sequential()
        for layer_config in self.model_config['layers']:
            if layer_config['type'] == 'lstm':
                layer_config['days_for_predict'] = self.days_for_predict
                layer_config['feature_num'] = self.feature_num
                break
        for layer_config in self.model_config['layers']:
            self.model.add(layer_dict[layer_config['type']](layer_config))
        if print_model:
            print(self.model.summary())
        self.model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=Adam(learning_rate=0.01),
            metrics=[MeanAbsolutePercentageError(), RootMeanSquaredError()]
        )

    def train(self, X, y, epochs, batch_size):
        callbacks = [
            EarlyStopping(monitor='loss', patience=5),  # Stop after 2 epochs whose loss is no longer decreasing
            # ModelCheckpoint(self.filename, monitor='loss', save_best_only=True)  # monitor is 'loss' not 'val_loss'
        ]
        print('\tEpochs: %s, Batch size: %s' % (epochs, batch_size))
        self.model.fit(
            X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=self.verbose, shuffle=False)
        # https://github.com/keras-team/keras/issues/2768
        print('\tModel saved as %s' % self.filename)

    def predict(self, X):
        return self.model.predict(X, batch_size=self.model_config['batch_size'])


def run_from_configs(data_config, model_config):
    print("------Implement on Stock Symbol '%s' and model '%s'-----" % (data_config["stock_code"], model_config["name"]))
    data = DataLoader(data_config)
    model = NNModel(data_config, model_config, data.get_columns_num())
    x_train, y_train, date_train, x_test, y_test, date_test, _ = data.get_windowed_data()
    return model.learn(
        x_train,
        y_train,
        x_test,
        y_test,
        date_test,
        data.get_min_max_scaler(),
        model_config['epochs'],
        model_config['batch_size']
    )


def run():
    config = load_config()
    stock_code_list = config['stock_code_list']
    res = {}
    data_config = config['data']
    for stock_code in stock_code_list:
        data_config['stock_code'] = stock_code
        for model_config in config['models']:
            if model_config['include'] is True:
                res[stock_code] = run_from_configs(data_config, model_config)
    for key in res:
        print(key, res[key])


if __name__ == '__main__':
    run()
