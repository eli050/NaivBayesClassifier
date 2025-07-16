from pprint import pprint
import pandas as pd
from services.data_cleaner.cleaner_service import CleanData
from services.data_reader.read_csv import ReadCSV
from math import log

class CreateModel:
    """Class for building a Naive Bayes-like model based on categorical probabilities"""


    def __init__(self,df:pd.DataFrame,target_column:str):
        """Initialize with a DataFrame and the name of the target column"""
        self.df = df
        self.target_column = target_column

    def get_dict_wights(self):
        """Creates a nested dictionary of log-probabilities for each feature value given a target class.
               - Calculates log-likelihoods with Laplace smoothing when needed
               - Also returns log-scaled prior probabilities for each target class
               Returns:
                   dict_wights: Nested dictionary [target][feature][value] -> log(probability)
                   dict_targets_size: Dictionary of prior log-probabilities per class
               """
        dict_targets_size = self.df[self.target_column].value_counts().to_dict()
        dict_wights = dict()
        for target in self.df[self.target_column].unique():
            dict_wights[target] = dict()
            dict_targets_size[target] = log(dict_targets_size[target] / self.df.size)
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
                            log((self.df[column][(self.df[self.target_column] == target)
                                        & (self.df[column] == value)].shape[0]) / vals_per_trg.value_counts().sum())
                else:
                    for value in values:
                        dict_wights[target][column][value] = \
                            log((self.df[column][(self.df[self.target_column] == target)
                                        & (self.df[column] == value)].shape[0] + 1 ) / (vals_per_trg.value_counts().sum() + values.size))
        return dict_wights , dict_targets_size






