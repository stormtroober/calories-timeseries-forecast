import pandas as pd
from preprocessing import preprocessing

dataframe = pd.read_csv('./dataset/Daily_activity_metrics.csv')

dataframe = preprocessing(dataframe, apply_scaling=False, apply_kalman=True, transform_method='box-cox')