# import libraries
import pandas as pd

# import dataset
data = pd.read_csv("../Arute_case/ds_exercise_data.csv")

# datetime processing
data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
data['day'] = data['Date'].dt.day
data['month'] = data['Date'].dt.month
data['year'] = data['Date'].dt.year

# fillna with mean of the month
data['CashIn'] = data['CashIn'].fillna(data.groupby(['year', 'month'])['CashIn'].transform('mean'))
data['CashOut'] = data['CashOut'].fillna(data.groupby(['year', 'month'])['CashOut'].transform('mean'))

print(data)
