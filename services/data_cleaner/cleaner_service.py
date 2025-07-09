from pandas import DataFrame

from services.data_reader.read_csv import ReadCSV


class CleanData:
    @staticmethod
    def clean_df(df:DataFrame):
        cleaner_df = df.dropna()
        cleaner_df = cleaner_df.drop(columns=['Name','PassengerId'])
        return cleaner_df, "Survived"

# cd = CleanData()
# df = cd.clean_df(ReadCSV("C:\\users\\home\\PycharmProjects\\NaiveBayesClassifier\\Data\\train.csv").get_data())[0]
# print(df.columns)