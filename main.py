import pandas as pd
import preprocessing

df = pd.read_csv("../Arute_case/ds_exercise_data.csv")
df = preprocessing.datetime_partitions(df)
df = preprocessing.fillna_with_mean(df)
train, test = preprocessing.train_test_split(df)
print(test.count())
