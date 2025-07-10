import pandas as pd
from services.data_cleaner.cleaner_service import CleanData
from services.data_reader.read_csv import ReadCSV
from services.model_trainer.create_model import CreateModel


class TestData:
    def __init__(self, model:dict,targets_size:dict):
        self.model = model
        self.targets_size = targets_size


    def get_grade(self,df:pd.DataFrame,target_column:str):
        list_targets = list(df[target_column])
        list_result = list()
        for column, value in df.iterrows():
            value = value.drop(target_column)
            list_result.append(self._check_target(value.to_dict()))
        matches = [item for item in list_result if item in list_targets]
        correct = (len(matches)/len(list_result))* 100
        return f"grad: {correct:.2f}%"



    def _check_target(self, values:dict):
        grades = dict()
        for target in self.model:
            grades[target] = 0
            for col, value in values.items():
                grades[target] += self.model[target][col][value]
        for key in self.targets_size:
            grades[key] += self.targets_size[key]
        max_targ = max(grades, key = grades.get)
        return max_targ







# cd = CleanData()
# df, target = cd.clean_df(ReadCSV("C:\\users\\home\\PycharmProjects\\NaiveBayesClassifier\\Data\\test.csv").get_data())
# df_test, target_test = cd.clean_df(ReadCSV("C:\\users\\home\\PycharmProjects\\NaiveBayesClassifier\\Data\\test.csv").get_data())
# cm = CreateModel(df,target)
# dw , ds = cm.get_dict_wights()
# td = TestData(dw,ds)
# print(td.get_grade(df_test, target_test))