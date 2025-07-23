import requests
from fastapi import FastAPI, APIRouter
from contextlib import asynccontextmanager


BASE_URL = "http://model_host:5001/model"


router_app = APIRouter()

ml_models = dict()




@asynccontextmanager
async def lifespan(app:FastAPI):
    try:
        response = requests.get(BASE_URL)
        ml_models.update(response.json())
    except Exception as e:
        print(e)
        raise
    yield


@router_app.get("/predict")
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
    for target in ml_models["model"]:
        grades[target] = 0
        for col, value in dict_data.items():
            try:
                grades[target] += ml_models["model"][target][col][value]
            except KeyError:
                grades[target] += sum(ml_models["model"][target][col].values()) / len(ml_models["model"][target][col])
    for key in ml_models["target_size"]:
        grades[key] += ml_models["target_size"][key]
    max_targ = max(grades, key=grades.get)
    return {"result" : int(max_targ)}
