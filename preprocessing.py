# import libraries
import pandas as pd
import datetime

# import dataset
data = pd.read_csv("../Arute_case/ds_exercise_data.csv")

# datetime processing
data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
data['day'] = data['Date'].dt.day
data['month'] = data['Date'].dt.month
data['year'] = data['Date'].dt.year

# fillna with mean of the month
# data['CashIn'] = data['CashIn'].fillna(data.groupby(['year', 'month'])['CashIn'].transform('mean'))
# data['CashOut'] = data['CashOut'].fillna(data.groupby(['year', 'month'])['CashOut'].transform('mean'))

# fillna with mean of the week
data['week_of_month'] = (data['day'] - 1) // 7 + 1
data['CashIn'] = data['CashIn'].fillna(data.groupby(['month', 'week_of_month'])['CashIn'].transform('mean'))
data['CashOut'] = data['CashOut'].fillna(data.groupby(['month', 'week_of_month'])['CashOut'].transform('mean'))

# print(data[data['CashOut']<=0])
# set Date column as index
data = data.set_index('Date')

# train and test split
train_df = data['2016-01-01':'2019-03-01']
test_df = data['2019-03-01':'2019-03-31']
