import pandas as pd


class ReadCSV:
    def __init__(self, file_path):
        self.file_path = file_path
    def get_data(self):
        df = pd.read_csv(self.file_path)
        return df



rc = ReadCSV("C:\\users\\home\\PycharmProjects\\NaiveBayesClassifier\\Data\\train.csv")
df = rc.get_data()
print(df.head())