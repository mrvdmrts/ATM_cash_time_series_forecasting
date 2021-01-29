import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.layers import Dropout


def scale_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df)
    return scaled


def train_test_split(df):
    """
    Splitting train and test data
    :param df: preprocessed dataframe
    :return: train and test dataframes
    """
    df = df.set_index('Date')
    train_df = df['2016-01-01':'2019-03-01']
    test_df = df['2019-03-01':'2019-03-31']
    train_df.reset_index(level='Date')
    test_df.reset_index(level='Date')
    return train_df['CashIn'], test_df['CashIn']


def LSTM(data):
    # data = df.iloc[:, 1:2].values
    # train_df, test_df = train_test_split(data[['Date', feature]])
    scaled_train = scale_data(data)
    features_set = []
    labels = []
    for i in range(60, len(scaled_train)):
        features_set.append(scaled_train[i - 60:i, 0])
        labels.append(scaled_train[i, 0])
    features_set, labels = np.array(features_set), np.array(labels)
    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, (features_set.shape[1], 1)))
