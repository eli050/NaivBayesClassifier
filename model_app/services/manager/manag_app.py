from services.data_cleaner.cleaner_service import CleanData
from services.data_loader.load_csv import LoadCSV
from services.evaluat_model.test_model import Evaluator
from services.model_trainer.train_model import Trainer
TRAIN_PATH = "././Data/train.csv"
TEST_PATH = "././Data/test.csv"


class ManagementApp:
    @staticmethod
    def start_flow():
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
