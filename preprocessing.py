# import libraries
import pandas as pd
import datetime


def datetime_partitions(df):
    """
    Day,month,year partition
    :param df:ds_exercise_data.csv
    :return: dataframe with day,month,year columns
    """
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    return df


def fillna_with_mean(df):
    """
    Filling null with mean of the week
    :param df: dataframe
    :return: dataframe
    """
    df['week_of_month'] = (df['day'] - 1) // 7 + 1
    df['CashIn'] = df['CashIn'].fillna(
        df.groupby(['month', 'week_of_month'])['CashIn'].transform('mean'))
    df['CashOut'] = df['CashOut'].fillna(
        df.groupby(['month', 'week_of_month'])['CashOut'].transform('mean'))
    return df


def train_test_split(df):
    """
    Splitting train and test
    :param df: preprocessed dataframe
    :return: train and test dataframes
    """
    df = df.set_index('Date')
    train_df = df['2016-01-01':'2019-03-01']
    test_df = df['2019-03-01':'2019-03-31']
    return train_df, test_df
