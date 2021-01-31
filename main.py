import pandas as pd
import preprocessing
import model
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("../Arute_case/ds_exercise_data.csv")
df = preprocessing.datetime_partitions(df)
df = preprocessing.add_holiday(df)
df = preprocessing.fillna_with_mean(df)

# CashIn
scaler = MinMaxScaler(feature_range=(0, 1))

prediction, actual = model.LSTM_model(df, 'CashIn', scaler)

result = pd.concat([pd.DataFrame(prediction[:, 0]), pd.DataFrame(scaler.inverse_transform(actual.reshape(1, -1))).T],
                   axis=1)
result.columns = ['predicted', 'actual']
result['diff'] = result['predicted'] - result['actual']

print(result)

model.plot(result['predicted'].round(2), result['actual'].round(2))
