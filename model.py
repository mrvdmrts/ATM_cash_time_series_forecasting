import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


def train_test_split(df, TRAIN_SIZE):
    train_size = int(len(df) * TRAIN_SIZE)
    test_size = len(df) - train_size
    train, test = df[0:train_size, :], df[train_size:len(df), :]
    return train, test


def create_dataset(df, window_size=1):
    data_X, data_Y = [], []
    for i in range(len(df) - window_size - 1):
        a = df[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(df[i + window_size, 0])
    return np.array(data_X), np.array(data_Y)


from tensorflow.keras.layers import Input


def fit_model(train_X, train_Y, window_size=1):
    model = Sequential()
    model.add(LSTM(10, input_shape=(1, window_size)))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="mse",
                  optimizer="adam"

                  )
    history = model.fit(train_X,
                        train_Y,
                        epochs=80,
                        batch_size=2,
                        verbose=2, shuffle=False)

    return model


def LSTM_model(df, feature, scaler,window_size):
    df = df[str(feature)]
    values = df.values.reshape(-1, 1)
    values = values.astype('float32')
    scaled_data = scaler.fit_transform(values)
    train, test = train_test_split(scaled_data, 0.6)
    train_X, train_Y = create_dataset(train, window_size=window_size)
    test_X, test_Y = create_dataset(test, window_size=window_size)
    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
    lstm_model = fit_model(train_X, train_Y, window_size=window_size)
    prediction = scaler.inverse_transform(lstm_model.predict(test_X))
    actual = scaler.inverse_transform(test_Y.reshape(1, -1))
    return prediction, actual


def plot(predict, actual):
    plt.plot(predict)
    plt.plot(actual)
    plt.title('final model ')
    plt.ylabel('prediction')
    plt.xlabel('actual')
    plt.legend(['pre', 'act'], loc='upper left')
    plt.show()
