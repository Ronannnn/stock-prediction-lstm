import datetime as dt
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, Bidirectional
from keras.models import Sequential, load_model

from model.data_processor import DataLoader
from model.model_abc import Model

# nn layer newer
from model.util import load_config, plot_pred_true_result


def new_dense(layer_config):
    neuron_num = layer_config['neuron_num'] if 'neuron_num' in layer_config else None
    input_dim = layer_config['input_dim'] if 'input_dim' in layer_config else None
    kernel_initializer = layer_config['kernel_initializer'] if 'input_dim' in layer_config else None
    activation = layer_config['activation'] if 'activation' in layer_config else None
    return Dense(neuron_num, input_dim=input_dim, kernel_initializer=kernel_initializer, activation=activation)


def new_lstm(layer_config):
    neuron_num, activation, return_seq, input_timesteps, input_dim = lstm_config(layer_config)
    if input_timesteps is None or input_dim is None:
        return LSTM(neuron_num, activation=activation, return_sequences=return_seq)
    else:
        return LSTM(
            neuron_num,
            activation=activation,
            input_shape=(input_timesteps, input_dim),
            return_sequences=return_seq
        )


def new_bi_lstm(layer_config):
    neuron_num, activation, return_seq, input_timesteps, input_dim = lstm_config(layer_config)
    if input_timesteps is None or input_dim is None:
        return Bidirectional(LSTM(neuron_num, activation=activation, return_sequences=return_seq))
    else:
        return Bidirectional(
            LSTM(neuron_num, activation=activation, return_sequences=return_seq),
            input_shape=(input_timesteps, input_dim)
        )


def lstm_config(layer_config):
    neuron_num = layer_config['neuron_num'] if 'neuron_num' in layer_config else None
    activation = layer_config['activation'] if 'activation' in layer_config else 'tanh'
    return_seq = layer_config['return_seq'] if 'return_seq' in layer_config else False
    input_timesteps = layer_config['input_timesteps'] if 'input_timesteps' in layer_config else None
    input_dim = layer_config['input_dim'] if 'input_dim' in layer_config else None
    return neuron_num, activation, return_seq, input_timesteps, input_dim


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
    "dense": new_dense,
    "lstm": new_lstm,
    "bilstm": new_bi_lstm,
    "dropout": new_dropout,
    "repeat": new_repeat_vector,
    "time_dense": new_time_distributed_dense
}


class NNModel(Model):

    def __init__(self, config, save_dir="saved_models", verbose=2):
        self.config = config
        self.filename = os.path.join(
            save_dir, '%s-%s.h5' % (self.config['name'], dt.datetime.now().strftime('%Y%m%d%H%M%S')))
        self.verbose = verbose

    def load(self, file):
        self.model = load_model(file)

    def build(self):
        self.model = Sequential()
        for layer_config in self.config['layers']:
            self.model.add(layer_dict[layer_config['type']](layer_config))
        self.model.compile(loss=self.config['loss'], optimizer=self.config['optimizer'])
        # forward_layer = LSTM(10, return_sequences=True)
        # backward_layer = LSTM(10, activation='relu', return_sequences=True, go_backwards=True)
        # self.model.add(Bidirectional(forward_layer, backward_layer=backward_layer, input_shape=(49, 4)))
        # self.model.add(Dense(1, activation='linear'))
        # self.model.compile(loss='mse', optimizer='adam')

    def train(self, X, y, epochs, batch_size):
        callbacks = [
            # EarlyStopping(monitor='loss', patience=2),  # Stop after 2 epochs whose loss is no longer decreasing
            # ModelCheckpoint(self.filename, monitor='loss', save_best_only=True)  # monitor is 'loss' not 'val_loss'
        ]
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=self.verbose)
        print('[Model] Training Completed. Model saved as %s' % self.filename)

    def predict(self, X):
        return self.model.predict(X, batch_size=self.config["batch_size"])


def nn_model_test():
    config = load_config()
    data = DataLoader(config["data"])
    for model_config in config['models']:
        if model_config['include'] is False:
            continue
        model = NNModel(model_config, config['data']['save_dir'], config['data']['verbose'])
        x_train, y_train, _ = data.get_windowed_train_data()
        x_pred, y_true, time_idx = data.get_windowed_test_data()
        y_pred = model.build_train_predict(x_train, y_train, x_pred, model_config["epochs"], model_config["batch_size"])
        model.evaluate(y_true.ravel(), y_pred.ravel())
        plot_pred_true_result(time_idx, y_pred.ravel(), y_true.ravel())
        # model.find_best_epoch(1, 40, 1, x_train, y_train, x_pred, y_true)
        # model.find_best_batch_size(0, 300, 10, x_train, y_train, x_pred, y_true)


if __name__ == '__main__':
    nn_model_test()