from pprint import pprint
import pandas as pd
from services.data_cleaner.cleaner_service import CleanData
from services.data_reader.read_csv import ReadCSV

class CreateModel:
    def __init__(self,df:pd.DataFrame,target_column:str):
        self.df = df
        self.target_column = target_column
    def get_dict_wights(self):
        dict_wights = dict()
        for target in self.df[self.target_column].unique():
            dict_wights[
                target
            ] = dict()
            for column in self.df.columns:
                if column == self.target_column:
                    continue
                dict_wights[target][column] = dict()
                values =  self.df[column].unique()
                vals_per_trg = self.df[column][(self.df[self.target_column] == target)]
                flag = set(values) == set(vals_per_trg.unique())
                if flag:
                    for value in values:
                        dict_wights[target][column][value] = \
                            (self.df[column][(self.df[self.target_column] == target)
                                        & (self.df[column] == value)].shape[0]) / vals_per_trg.value_counts().sum()
                else:
                    for value in values:
                        dict_wights[target][column][value] = \
                            (self.df[column][(self.df[self.target_column] == target)
                                        & (self.df[column] == value)].shape[0] + 1 ) / (vals_per_trg.value_counts().sum() + values.size)
        return dict_wights






# rc = ReadCSV("C:\\users\\home\\PycharmProjects\\NaiveBayesClassifier\\Data\\train.csv")
# df_trg = CleanData().clean_df(rc.get_data())
# pprint(df_trg[0].columns)
# cm = CreateModel(df_trg[0],df_trg[1])
# pprint(cm.get_dict_wights()[0].keys())
# cm.get_dict_wights()