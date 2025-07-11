import pandas as pd
from services.data_cleaner.cleaner_service import CleanData
from services.data_reader.read_csv import ReadCSV
from services.model_trainer.test_model import TestData
from services.model_trainer.create_model import CreateModel


class Menu:
    def __init__(self):
        df = ReadCSV("C:\\users\\home\\PycharmProjects\\NaiveBayesClassifier\\Data\\train.csv").get_data()
        self.cleaned_df, self.target = CleanData.clean_df(df)
        self.model, self.target_size = CreateModel(self.cleaned_df, self.target).get_dict_wights()
        self.features = [col for col in self.cleaned_df.columns if col != self.target]

    def run(self):
        while True:
            print("\nChoose an option:")
            print("1. Check model accuracy")
            print("2. Predict survival based on user input")
            print("3. Exit")
            choice = input("Enter your choice: ")

            if choice == "1":
                print(TestData(self.model, self.target_size).get_grade(self.cleaned_df, self.target))
            elif choice == "2":
                self.predict_individual()
            elif choice == "3":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Try again.")

    def predict_individual(self):
        user_data = dict()
        for col in self.features:
            values = self.cleaned_df[col].dropna().unique()
            if self.cleaned_df[col].dtype == 'O' or len(values) < 20:
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
            else:
                while True:
                    try:
                        val = float(input(f"Enter a numeric value for '{col}': "))
                        user_data[col] = val
                        break
                    except ValueError:
                        print("Invalid number. Try again.")

        result = self._check_target(user_data)
        prediction = max(result, key=result.get)
        print(f"\nPrediction: {prediction}")

    def _check_target(self, values: dict):
        grades = dict()
        for target in self.model:
            grades[target] = 0
            for feature in values:
                if values[feature] in self.model[target][feature]:
                    grades[target] += self.model[target][feature][values[feature]]
        return grades


# menu = Menu()
# menu.run()