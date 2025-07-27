from services.data_cleaner.cleaner_service import CleanData
from services.data_loader.load_csv import LoadCSV
from services.model_trainer.train_model import Trainer
from classifier_terminal import ClassifierMenu

if __name__ == '__main__':
    df = LoadCSV("C:\\Users\\HOME\\PycharmProjects\\NaiveBayesClassifier\\model_app\\Data\\train.csv").get_data()
    cleaned_df, target = CleanData.clean_df(df)
    model, target_size = Trainer(cleaned_df, target).train_model()
    df = LoadCSV("C:\\Users\\HOME\\PycharmProjects\\NaiveBayesClassifier\\model_app\\Data\\test.csv").get_data()
    test_df ,test_targ = CleanData.clean_df(df)


    menu = ClassifierMenu(cleaned_df, test_df, target, model, target_size)
    menu.run()