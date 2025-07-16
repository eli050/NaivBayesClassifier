import pandas as pd
from services.data_cleaner.cleaner_service import CleanData
from services.data_reader.read_csv import ReadCSV
from services.model_trainer.test_model import TestData
from services.model_trainer.create_model import CreateModel


class Menu:
    """Class for displaying a terminal menu to interact with the model"""

    def __init__(self, cleaned_df:pd.DataFrame, test_df:pd.DataFrame, target:str, model:dict, target_size:dict):
        """Initialize the menu with training and test data, model, and target details"""
        self.cleaned_df = cleaned_df
        self.target = target
        self.model = model
        self.target_size = target_size
        self.test_df = test_df
        self.features = [col for col in self.cleaned_df.columns if col != self.target]

    def run(self):
        """Runs the main menu loop, providing options to check model accuracy or predict a single input"""
        while True:
            print("\nChoose an option:")
            print("1. Check model accuracy")
            print("2. Predict survival based on user input")
            print("3. Exit")
            choice = input("Enter your choice: ")

            if choice == "1":
                print(TestData(self.model, self.target_size).get_grade(self.test_df, self.target))
            elif choice == "2":
                prediction = self.predict_individual()
                print(f"\nPrediction: {prediction}")
            elif choice == "3":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Try again.")

    def predict_individual(self) -> str:
        """Prompts the user to enter feature values, then predicts the target based on the input"""
        user_data = dict()
        for col in self.features:
            values = self.cleaned_df[col].unique()
            values = sorted(set(values))
            print(f"\nSelect a value for '{col}':")
            for i, val in enumerate(values):
                print(f"{i + 1}. {val}")
            while True:
                try:
                    idx = int(input("Enter your choice: "))
                    if 1 <= idx <= len(values):
                        user_data[col] = values[idx - 1]
                        break
                    else:
                        print("Choice out of range.")
                except ValueError:
                    print("Please enter a valid number.")

        print(user_data)
        result = self._check_target(user_data)
        return result

    def _check_target(self, values: dict) -> str:
        """Predicts the target class for the given input dictionary using the model"""

        grades = dict()
        for target in self.model:
            grades[target] = 0
            for col, value in values.items():
                try:
                    grades[target] += self.model[target][col][value]
                except KeyError:
                    grades[target] += sum(self.model[target][col].values()) / len(self.model[target][col])
        for key in self.target_size:
            grades[key] += self.target_size[key]
        max_targ = max(grades, key=grades.get)
        return max_targ

