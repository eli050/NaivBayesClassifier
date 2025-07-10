import pandas as pd
from pandas import DataFrame
from pprint import pprint

df = pd.read_csv("Data/train.csv")
df = df.dropna()
df = df.drop(columns=['Name','PassengerId'])

dict_wights = dict()
for target in df["Survived"].unique():
    dict_wights[
        target
    ] = dict()
    for column in df.columns[1:]:
        dict_wights[target][column] = dict()
        for value in df[column].unique():
            dict_wights[target][column][value] = df[column][(df["Survived"] == target) & (df[column] == value)].shape[0]


pprint(dict_wights)






class BaseWights:
    @staticmethod
    def get_wights(df:DataFrame):
        pass