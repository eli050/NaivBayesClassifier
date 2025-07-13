import pandas as pd
import numpy as np

from services.data_reader.read_csv import ReadCSV


class CleanData:
    """Class for cleaning and preprocessing a Titanic dataset"""


    @staticmethod
    def clean_df(df:pd.DataFrame):
        """Cleans the DataFrame by:
              - Dropping missing values and irrelevant columns
              - Binning 'Age' into ranges
              - Binning 'Fare' into categories
              - Replacing 'Embarked' codes with full port names
              Returns the cleaned DataFrame and the target column name
              """
        cleaner_df = df.dropna()
        cleaner_df = cleaner_df.drop(columns=['Name','PassengerId','Ticket','Cabin'])
        conditions = [
            cleaner_df['Age'] < 20,
            cleaner_df['Age'].between(20,39),
            cleaner_df['Age'].between(40, 59),
            cleaner_df['Age'].between(60, 80)
        ]
        choices = ["< 20","20-39","40-59","60-80"]
        cleaner_df['Age'] = np.select(conditions,choices,default="")
        conditions = [
            cleaner_df['Fare'] < 30,
            cleaner_df['Fare'].between(30, 150),
            cleaner_df['Fare'] > 150
        ]
        choices = ['Cheap', 'Average', 'Expensive']
        cleaner_df['Fare'] = np.select(conditions, choices, default="")
        cleaner_df['Embarked'] = cleaner_df['Embarked'].replace({
            'C': 'Cherbourg',
            'Q': 'Queenstown',
            'S': 'Southampton'

        })

        return cleaner_df, "Survived"

# cd = CleanData()
# df = cd.clean_df(ReadCSV("C:\\users\\home\\PycharmProjects\\NaiveBayesClassifier\\Data\\train.csv").get_data())[0]
# print(df["SibSp"].max())
# print(df['Embarked'])