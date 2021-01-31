import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

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
    return train_df,test_df


def LSTM(df,feature):
    data=df['Date',feature]
    train,test=train_test_split(data)
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train)
    scaled_test = scaler.transform(test)
    n_input = 365
    n_features = 1
    train_generator = TimeseriesGenerator(scaled_train,
                                          scaled_train,
                                          n_input,
                                          batch_size=1)
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features), return_sequences=True))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(LSTM(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    model.fit(train_generator, epochs=30)