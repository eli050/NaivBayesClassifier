from app.cleaner import CleanData
from app.loader import LoadCSV
from app.test_model import Evaluator
from app.train_model import Trainer
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


TRAIN_PATH = f"{PROJECT_ROOT}/Data/train.csv"
TEST_PATH = f"{PROJECT_ROOT}/Data/test.csv"


class ManagementApp:
    """
        ManagementApp class orchestrates the full machine learning flow,
        from loading and cleaning the data to training the model and evaluating it.
    """
    @staticmethod
    def start_flow():
        """
               Executes the complete ML pipeline:

               1. Loads the training data.
               2. Cleans the training data and separates features and target.
               3. Trains a model using the cleaned data.
               4. Loads and cleans the test data.
               5. Evaluates the model on the test set.
               6. Returns the model, target size (class weights), and evaluation grade.
        """
        lc = LoadCSV(TRAIN_PATH)
        data_raw = lc.get_data()
        cd = CleanData()
        df, target = cd.clean_df(data_raw)
        train = Trainer(df, target)
        lc = LoadCSV(TEST_PATH)
        test_data = lc.get_data()
        test_df, target = cd.clean_df(test_data)
        model, target_size = train.train_model()
        ev = Evaluator(model, target_size)
        grade = ev.get_grade(test_df,target)
        return {"model":model,"target_size":target_size,"grade":grade}
