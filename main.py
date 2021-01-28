import pandas as pd
import preprocessing
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

df = pd.read_csv("../Arute_case/ds_exercise_data.csv")
df = preprocessing.datetime_partitions(df)
df = preprocessing.add_holiday(df)
df = preprocessing.fillna_with_mean(df)


print(df)

# train, test = preprocessing.train_test_split(df)
