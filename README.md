# USC DSCI 560 Project - Stock Prediction

[LSTM example](https://www.altumintelligence.com/articles/a/Time-Series-Prediction-Using-LSTM-Deep-Neural-Networks)

[LSTM detailed](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

[Sliding(Rolling) windows Analysis of Time-Series model](https://www.mathworks.com/help/econ/rolling-window-estimation-of-state-space-models.html)

[Time Series Data Prediction Using Sliding Window Based RBF Neural Network](https://www.ripublication.com/ijcir17/ijcirv13n5_46.pdf)
1. Data pre-processing
    1. data smoothing
    2. feature extraction
    3. feature selection

#### Stock Codes
http://vip.stock.finance.sina.com.cn/usstock/ustotal.php
Google GOOGL

#### Extra:
1. dim == features i.e. number of columns
2. time steps == sequence length - 1 since the last one is used as y

## Python

### slice
[Reference](https://www.pythoninformer.com/python-libraries/numpy/index-and-slice/)

## Model Reference
[Linear SVM first Reference](https://github.com/chaitjo/regression-stock-prediction/blob/master/svr.py)
[Encoder Decoder LSTM](https://deepdatainsight.com/stock-market-prediction-approaches/)