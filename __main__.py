from services.data_cleaner.cleaner_service import CleanData
from services.data_reader.read_csv import ReadCSV
from services.model_trainer.create_model import CreateModel
from services.manag_ui.ui import Menu

if __name__ == '__main__':
    df = ReadCSV("C:\\users\\home\\PycharmProjects\\NaiveBayesClassifier\\Data\\train.csv").get_data()
    cleaned_df, target = CleanData.clean_df(df)
    model, target_size = CreateModel(cleaned_df, target).get_dict_wights()
    df = ReadCSV("C:\\users\\home\\PycharmProjects\\NaiveBayesClassifier\\Data\\test.csv").get_data()
    test_df ,test_targ = CleanData.clean_df(df)


    menu = Menu(cleaned_df,test_df,target,model,target_size)
    menu.run()