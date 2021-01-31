# import libraries
import pandas as pd
import datetime


def datetime_partitions(df):
    """
    :param df:ds_exercise_data.csv
    :return: dataframe with day,month,year columns
    """
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    # It determines the week of a day in a month
    DAYS_IN_A_WEEK = 7
    df['week_of_month'] = (df['day'] - 1) // DAYS_IN_A_WEEK + 1
    return df


def fillna_with_mean(df):
    """
        It fills null values with average cash in and cash out values from same days of the other years
        :param df: dataframe
        :return: dataframe
    """
    #CashIn
    month = df.iloc[(df['CashIn'].isnull()).values, df.columns.get_indexer(['month'])].values.tolist()
    day = df.iloc[(df['CashIn'].isnull()).values, df.columns.get_indexer(['day'])].values.tolist()
    mean = []
    for i in range(0, len(month)):
        mean.append(
            df.loc[((df.day == day[i][0]) & (df.month == month[i][0]) & (df.CashIn.notnull())), 'CashIn'].mean())

    df.loc[(df['CashIn'].isnull()), 'CashIn'] = mean
    #CashOut
    month = df.iloc[(df['CashOut'].isnull()).values, df.columns.get_indexer(['month'])].values.tolist()
    day = df.iloc[(df['CashOut'].isnull()).values, df.columns.get_indexer(['day'])].values.tolist()
    mean = []
    for i in range(0, len(month)):
        mean.append(
            df.loc[((df.day == day[i][0]) & (df.month == month[i][0]) & (df.CashIn.notnull())), 'CashOut'].mean())

    df.loc[(df['CashOut'].isnull()), 'CashOut'] = mean
    return df

def add_holiday(df):
    """
    :param df: pandas dataframe
    """
    df['holiday'] = 0
    df.loc[((df['month'] == 1) & (df['day'] == 1)) |
           ((df['month'] == 4) & (df['day'] == 23)) |
           ((df['month'] == 5) & (df['day'] == 1)) |
           ((df['month'] == 5) & (df['day'] == 19)) |
           ((df['month'] == 8) & (df['day'] == 30)) |
           ((df['month'] == 10) & (df['day'] == 28)) |
           ((df['month'] == 10) & (df['day'] == 29))
    , 'holiday'] = 1
    return  df
