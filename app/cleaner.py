import pandas as pd
import numpy as np


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
        cleaner_df = df.dropna().copy(deep=True)
        cleaner_df = cleaner_df.drop(columns=['Name','PassengerId','Ticket','Cabin'])
        conditions = [
            cleaner_df['Age'] < 20,
            cleaner_df['Age'].between(20,39),
            cleaner_df['Age'].between(40, 59),
            cleaner_df['Age'].between(60, 80)
        ]
        choices = ["0-20","20-39","40-59","60-80"]
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
        cleaner_df = cleaner_df.map(lambda x: str(x) if not isinstance(x, str) else x)
        return cleaner_df, "Survived"

