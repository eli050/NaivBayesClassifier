from fastapi import APIRouter
from services.data_cleaner.cleaner_service import CleanData
from services.data_loader.load_csv import LoadCSV
from services.model_trainer.train_model import Trainer

router = APIRouter()

CD = CleanData()
DF, TARGET = CD.clean_df(LoadCSV("././Data/train.csv").get_data())
MODEL, TARGET_SIZE = Trainer(DF, TARGET).train_model()




@router.get("/predict")
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
    for target in MODEL:
        grades[target] = 0
        for col, value in dict_data.items():
            try:
                grades[target] += MODEL[target][col][value]
            except KeyError:
                grades[target] += sum(MODEL[target][col].values()) / len(MODEL[target][col])
    for key in TARGET_SIZE:
        grades[key] += TARGET_SIZE[key]
    max_targ = max(grades, key=grades.get)
    return {"result" : int(max_targ)}



