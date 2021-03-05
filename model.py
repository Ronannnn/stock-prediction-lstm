import datetime as dt
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential, load_model

from utils import Timer


class Model:

    def load(self, file):
        print('[Model] Loading model from file %s' % file)
        self.model = load_model(file)

    def build(self, config):
        self.filename_with_dt = '%s-%s' % (config['name'], dt.datetime.now().strftime('%Y%m%d%H%M%S'))
        timer = Timer()
        self.model = Sequential()
        for layer in config['layers']:
            neurons = layer['neuron_num'] if 'neuron_num' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                if input_timesteps is not None and input_dim is not None:
                    self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
                else:
                    self.model.add(LSTM(neurons, return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=config['loss'], optimizer=config['optimizer'])

        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir):
        timer = Timer()
        save_filename = os.path.join(save_dir, '%s-e%s.h5' % (self.filename_with_dt, str(epochs)))
        callbacks = [
            EarlyStopping(monitor='loss', patience=2),  # Stop after 2 epochs whose loss is no longer decreasing
            ModelCheckpoint(filepath=save_filename, monitor='loss', save_best_only=True)  # monitor is 'loss' not 'val_loss'
        ]
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2)
        print('[Model] Training Completed. Model saved as %s' % save_filename)
        timer.stop()

    def train_generator(self, data_generator, epochs, batch_size, steps_per_epoch, save_dir):
        timer = Timer()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))

        save_filename = os.path.join(save_dir, '%s-e%s.h5' % (self.filename_with_dt, str(epochs)))
        callbacks = [
            ModelCheckpoint(
                filepath=save_filename,
                monitor='loss',
                save_best_only=True
            )
        ]
        self.model.fit_generator(
            data_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1,
            verbose=2
        )

        print('[Model] Training Completed. Model saved as %s' % save_filename)
        timer.stop()

    def predict(self, data, batch_size):
        # Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting...')
        return self.model.predict(data, batch_size=batch_size)
