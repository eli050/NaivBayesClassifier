import uvicorn
from fastapi import FastAPI

from services.data_cleaner.cleaner_service import CleanData
from services.data_reader.read_csv import ReadCSV
from services.model_trainer.create_model import CreateModel

app = FastAPI()

CD = CleanData()
df, target = CD.clean_df(ReadCSV("C:\\users\\home\\PycharmProjects\\NaiveBayesClassifier\\Data\\train.csv").get_data())
model, target_size = CreateModel(df, target).get_dict_wights()


@app.get("/predict")
def get_grade(
    Pclass: int,
    Sex: str,
    Age: str,
    SibSp: int,
    Parch: int,
    Fare: str,
    Embarked: str
):
    grades = dict()
    dict_data = {
        "Pclass": Pclass,
        "Sex": Sex,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Embarked": Embarked
    }
    for target in model:
        grades[target] = 0
        for col, value in dict_data.items():
            try:
                grades[target] += model[target][col][value]
            except KeyError:
                grades[target] += sum(model[target][col].values()) / len(model[target][col])
    for key in target_size:
        grades[key] += target_size[key]
    max_targ = max(grades, key=grades.get)
    return {"result" : int(max_targ)}

if __name__ == '__main__':
    uvicorn.run(app)

