import sys
import os
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
from utils import load_config
import time
import sklearn
import math
#from data_processor import DataLoader.fetch_senti_score
class Model:
    def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        output_size,
        forget_bias = 0.1,
    ):
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple = False)

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)],
            state_is_tuple = False,
        )
        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, output_size))
        drop = tf.contrib.rnn.DropoutWrapper(
            rnn_cells, output_keep_prob = forget_bias
        )
        self.hidden_layer = tf.placeholder(
            tf.float32, (None, num_layers * 2 * size_layer)
        )
        _, last_state = tf.nn.dynamic_rnn(
            drop, self.X, initial_state = self.hidden_layer, dtype = tf.float32
        )
        
        with tf.variable_scope('decoder', reuse = False):
            rnn_cells_dec = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell(size_layer) for _ in range(num_layers)], state_is_tuple = False
            )
            drop_dec = tf.contrib.rnn.DropoutWrapper(
                rnn_cells_dec, output_keep_prob = forget_bias
            )
            self.outputs, self.last_state = tf.nn.dynamic_rnn(
                drop_dec, self.X, initial_state = last_state, dtype = tf.float32
            )
            
        self.logits = tf.layers.dense(self.outputs[-1], output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )
        
def calculate_accuracy(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100
def cal_rmse(real, predict):
    mse = sklearn.metrics.mean_squared_error(real, predict)
    rmse = math.sqrt(mse)
    return rmse
def cal_r_square(real, predict):
    r2 = sklearn.metrics.r2_score(real, predict)
    return r2
def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer

def forecast():
    tf.reset_default_graph()
    modelnn = Model(
        learning_rate, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate
    )
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()

    pbar = tqdm(range(epoch), desc = 'train loop')
    for i in pbar:
        init_value = np.zeros((1, num_layers * 2 * size_layer))
        total_loss, total_acc = [], []
        for k in range(0, df_train.shape[0] - 1, timestamp):
            index = min(k + timestamp, df_train.shape[0] - 1)
            batch_x = np.expand_dims(
                df_train.iloc[k : index, :].values, axis = 0
            )
            batch_y = df_train.iloc[k + 1 : index + 1, :].values
            logits, last_state, _, loss = sess.run(
                [modelnn.logits, modelnn.last_state, modelnn.optimizer, modelnn.cost],
                feed_dict = {
                    modelnn.X: batch_x,
                    modelnn.Y: batch_y,
                    modelnn.hidden_layer: init_value,
                },
            )        
            init_value = last_state
            total_loss.append(loss)
            total_acc.append(calculate_accuracy(batch_y[:, 0], logits[:, 0]))
        pbar.set_postfix(cost = np.mean(total_loss), acc = np.mean(total_acc))
    
    future_day = test_size

    output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
    output_predict[0] = df_train.iloc[0]
    upper_b = (df_train.shape[0] // timestamp) * timestamp
    init_value = np.zeros((1, num_layers * 2 * size_layer))

    for k in range(0, (df_train.shape[0] // timestamp) * timestamp, timestamp):
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict = {
                modelnn.X: np.expand_dims(
                    df_train.iloc[k : k + timestamp], axis = 0
                ),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[k + 1 : k + timestamp + 1] = out_logits

    if upper_b != df_train.shape[0]:
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict = {
                modelnn.X: np.expand_dims(df_train.iloc[upper_b:], axis = 0),
                modelnn.hidden_layer: init_value,
            },
        )
        output_predict[upper_b + 1 : df_train.shape[0] + 1] = out_logits
        future_day -= 1
        date_ori.append(date_ori[-1] + timedelta(days = 1))

    init_value = last_state
    
    for i in range(future_day):
        o = output_predict[-future_day - timestamp + i:-future_day + i]
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict = {
                modelnn.X: np.expand_dims(o, axis = 0),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[-future_day + i] = out_logits[-1]
        date_ori.append(date_ori[-1] + timedelta(days = 1))
    #print('111',output_predict)
    #print(len(output_predict))
    output_predict = minmax.inverse_transform(output_predict)
    #print('222',output_predict)
    deep_future = anchor(output_predict[:, 0], 0.3)
    
    return deep_future[-test_size:]
def fetch_senti_score(stock_code):
    """
    Sentiment Analysis Score
    :return:
    """
    senti_data_dir ="model/data/sentiment/"
    filename = "%s%s.csv" % (senti_data_dir, stock_code.upper())
    if not os.path.exists(filename):
        return pd.DataFrame()
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index, format='%d-%b-%y').strftime('%Y-%m-%d')
    return df
#time start
start_time = time.time()
#sns.set()
#tf.compat.v1.random.set_random_seed(1234)
config = load_config()
data_config = config['data']
start = data_config["start"]
end = data_config["end"]
for stock_code in data_config['stock_code_list']:
    print("---------------")
    print("Stock Code: %s" % stock_code)
    data_config['stock_code'] = stock_code
    #data = DataLoader(data_config)
    #stock_code = config['data']["stock_code"]
    senti_df = fetch_senti_score(stock_code)
    df = yf.download(stock_code,start=start,end=end)
    
    if not senti_df.empty:
        df = df.merge(senti_df,  left_index=True, right_index=True)
        print(df)
    df.insert(0, 'index', range(len(df)))
    df['Date'] = df.index
    df.set_index(['index'],inplace=True)
    if 'score' in df:
        df = df[['Date','Open','High','Low','Close','Adj Close','Volume','score']]
    else:
        df = df[['Date','Open','High','Low','Close','Adj Close','Volume']]
    #df.head()
    minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32')) # Close index
    df_log = minmax.transform(df.iloc[:, 4:5].astype('float32')) # Close index
    df_log = pd.DataFrame(df_log)
    #df_log.head()
    train_test_split = data_config['train_test_split']
    lendf = len(df)
    test_size = int((1-train_test_split)*lendf)
    df_train = df_log.iloc[:-test_size]
    df_test = df_log.iloc[-test_size:]
    future_day = test_size
    for model_config in config['models']:
        if model_config['name'] == "seq2seq_model":
            #df.shape, df_train.shape, df_test.shape
            simulation_size = model_config["simulation_size"]
            num_layers = model_config['num_layers']
            size_layer = model_config['batch_size']
            timestamp = model_config['timestamp']
            epoch = model_config['epochs']
            dropout_rate = model_config['dropout_rate']
            learning_rate = model_config['learning_rate']
    results = []
    for i in range(simulation_size):
        print('simulation %d'%(i + 1))
        results.append(forecast())

    accuracies = [calculate_accuracy(df['Close'].iloc[-test_size:].values, r) for r in results]
    rmse = []
    rmse = [cal_rmse(df['Close'].iloc[-test_size:].values, r) for r in results]
    RMSE = min(rmse)
    print('RMSE:',RMSE)
    r2 = []
    r2 = [cal_r_square(df['Close'].iloc[-test_size:].values, r) for r in results]
    r_square= max(r2)
    print('r_square:',r_square)
    plt.figure(figsize = (15, 5))
    for no, r in enumerate(results):
        plt.plot(r, label = 'forecast %d'%(no + 1))
    plt.plot(df['Close'].iloc[-test_size:].values, label = 'true trend', c = 'black')
    plt.legend()
    plt.title('average accuracy: %.4f'%(np.mean(accuracies)))
    stock_name = stock_code
    plt.savefig('saved_plot/'+stock_name + '.png')
    #plt.show()
    end_time = time.time()
    total_time = end_time-start_time
    print('total_time:',total_time)
    import csv
    with open('Result_before_pademic.csv', mode='a+',newline='') as csv_file:
        csv_write = csv.writer(csv_file)
        csv_write.writerow([stock_code,start,end,RMSE,r_square,total_time])