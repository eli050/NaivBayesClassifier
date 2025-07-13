import pandas as pd
from services.data_cleaner.cleaner_service import CleanData
from services.data_reader.read_csv import ReadCSV
from services.model_trainer.create_model import CreateModel



class TestData:
    """Class for testing data against a trained probabilistic model"""


    def __init__(self, model:dict,targets_size:dict):
        """Initialize with the trained model and log-scaled prior probabilities"""
        self.model = model
        self.targets_size = targets_size


    def get_grade(self,df:pd.DataFrame,target_column:str) -> str:
        """Calculates the prediction accuracy of the model on the provided test DataFrame.
               Compares predicted targets with actual targets and returns the accuracy percentage.

               Args:
                   df: DataFrame to test
                   target_column: Name of the target column in the DataFrame

               Returns:
                   A string with the prediction accuracy as a percentage
               """
        list_targets = list(df[target_column])
        list_result = list()
        for column, value in df.iterrows():
            value = value.drop(target_column)
            list_result.append(self._check_target(value.to_dict()))
        sum_correct = sum(pred == true
                      for pred, true in zip(list_result, list_targets))
        correct = (sum_correct/len(list_result))* 100
        return f"grad: {correct:.2f}%"



    def _check_target(self, values:dict) -> str:
        """Predicts the target class based on the input feature dictionary.
               Calculates total score for each target and selects the one with the highest score.
               Uses average smoothing for unknown feature values.

               Args:
                   values: Dictionary of feature values for a single sample

               Returns:
                   The predicted target class
               """
        grades = dict()
        for target in self.model:
            grades[target] = 0
            for col, value in values.items():
                try:
                    grades[target] += self.model[target][col][value]
                except KeyError:
                    grades[target] +=  sum(self.model[target][col].values()) / len(self.model[target][col])
        for key in self.targets_size:
            grades[key] += self.targets_size[key]
        max_targ = max(grades, key = grades.get)
        return max_targ







# cd = CleanData()
# df, target = cd.clean_df(ReadCSV("C:\\users\\home\\PycharmProjects\\NaiveBayesClassifier\\Data\\train.csv").get_data())
# df_test, target_test = cd.clean_df(ReadCSV("C:\\users\\home\\PycharmProjects\\NaiveBayesClassifier\\Data\\test.csv").get_data())
# cm = CreateModel(df,target)
# dw , ds = cm.get_dict_wights()
# td = TestData(dw,ds)
# print(td.get_grade(df_test, target_test))