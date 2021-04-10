import datetime as dt
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential, load_model

from model.data_processor import DataLoader
from model.model_abc import Model

# nn layer newer
from model.util import load_config, plot_result


def new_dense(layer_config):
    neuron_num = layer_config['neuron_num'] if 'neuron_num' in layer_config else None
    input_dim = layer_config['input_dim'] if 'input_dim' in layer_config else None
    kernel_initializer = layer_config['kernel_initializer'] if 'input_dim' in layer_config else None
    activation = layer_config['activation'] if 'activation' in layer_config else None
    return Dense(neuron_num, input_dim=input_dim, kernel_initializer=kernel_initializer, activation=activation)


def new_lstm(layer_config):
    neuron_num = layer_config['neuron_num'] if 'neuron_num' in layer_config else None
    input_timesteps = layer_config['input_timesteps'] if 'input_timesteps' in layer_config else None
    input_dim = layer_config['input_dim'] if 'input_dim' in layer_config else None
    return_seq = layer_config['return_seq'] if 'return_seq' in layer_config else None
    input_shape = (input_timesteps, input_dim) if input_timesteps is not None and input_dim is not None else None
    return LSTM(neuron_num, input_shape=input_shape, return_sequences=return_seq)


def new_dropout(layer_config):
    dropout_rate = layer_config['rate'] if 'rate' in layer_config else None
    return Dropout(dropout_rate)


layer_dict = {
    "dense": new_dense,
    "lstm": new_lstm,
    "dropout": new_dropout
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

    def train(self, x, y):
        callbacks = [
            EarlyStopping(monitor='loss', patience=2),  # Stop after 2 epochs whose loss is no longer decreasing
            ModelCheckpoint(self.filename, monitor='loss', save_best_only=True)  # monitor is 'loss' not 'val_loss'
        ]
        epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=self.verbose)
        print('[Model] Training Completed. Model saved as %s' % self.filename)

    def train_with_generator(self, data_generator):
        epochs = self.config["epochs"]
        steps_per_epoch = self.config["steps_per_epoch"]
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batches per epoch' % (epochs, steps_per_epoch))
        callbacks = [
            ModelCheckpoint(self.filename, monitor='loss', save_best_only=True)
        ]
        self.model.fit_generator(
            data_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1,
            verbose=self.verbose
        )
        print('[Model] Training Completed. Model saved as %s' % self.filename)

    def predict(self, x):
        return self.model.predict(x, batch_size=self.config["batch_size"])


def nn_model_test():
    config = load_config()
    data = DataLoader(config["data"])
    for model_config in config['models']:
        if model_config['include'] is False:
            continue
        model = NNModel(model_config, config['data']['save_dir'], config['data']['verbose'])
        # get data
        x_train, y_train, _ = data.get_windowed_train_data()
        x_predict, y_true, time_idx = data.get_windowed_test_data()
        # feed in model and get prediction
        y_predict = model.build_train_predict(x_train, y_train, x_predict)
        plot_result(y_predict, y_true, time_idx)


if __name__ == '__main__':
    nn_model_test()
