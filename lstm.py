from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM


def scale_data(train_df, test_df):
    scaler = MinMaxScaler()
    scaler.fit(train_df)
    scaled_train_data = scaler.transform(train_df)
    scaled_test_data = scaler.transform(test_df)
    return scaled_train_data, scaled_test_data


def train_test_split(df):
    """
    Splitting train and test data
    :param df: preprocessed dataframe
    :return: train and test dataframes
    """
    df.index.freq = 'D'
    df = df.set_index('Date')
    train_df = df['2016-01-01':'2019-03-01']
    test_df = df['2019-03-01':'2019-03-31']
    return train_df, test_df


def LSTM(df, feature):
    train_df, test_df = train_test_split(df[['Date', feature]])
    scaled_train, scaled_test = scale_data(train_df, test_df)
    n_input = 365
    n_features = 1
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

    lstm_model = Sequential()
    lstm_model.add(LSTM(200, input_shape=(n_input, n_features)))
    lstm_model.add(Activation('relu'))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.summary()


def plot_loss_history(model):
    losses = model.history.history['loss']
    plt.figure(figsize=(12, 4))
    plt.xticks(np.arange(0, 21, 1))
    plt.plot(range(len(losses)), losses)
    plt.show()


def predictions():
    lstm_predictions_scaled = list()

    batch = scaled_train_data[-n_input:]
    current_batch = batch.reshape((1, n_input, n_features))

    for i in range(len(test_df)):
        lstm_pred = lstm_model.predict(current_batch)[0]
        lstm_predictions_scaled.append(lstm_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[lstm_pred]], axis=1)
