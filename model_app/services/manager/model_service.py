from fastapi import APIRouter,FastAPI
from contextlib import asynccontextmanager
from services.manager.manag_app import ManagementApp
from pprint import pprint

router = APIRouter()



ml_models = dict()

@asynccontextmanager
async def lifespan(app:FastAPI):
    try:
        ml_models.update(ManagementApp.start_flow())
    except Exception as e:
        ml_models["error"] = e
    yield



@router.get("/model")
def get_model():
    if "error" not in ml_models:
        print(ml_models["grade"])
        return {"model":ml_models["model"], "target_size":ml_models["target_size"]}
    else:
        return {"error":"The model is not yet trained."}




# @router.get("/predict")
# def get_grade(
#     Pclass: int,
#     Sex: str,
#     Age: str,
#     SibSp: int,
#     Parch: int,
#     Fare: str,
#     Embarked: str
# ):
#     grades = dict()
#     dict_data = {
#         "Pclass": Pclass,
#         "Sex": Sex,
#         "Age": Age,
#         "SibSp": SibSp,
#         "Parch": Parch,
#         "Fare": Fare,
#         "Embarked": Embarked
#     }
#     for target in MODEL:
#         grades[target] = 0
#         for col, value in dict_data.items():
#             try:
#                 grades[target] += MODEL[target][col][value]
#             except KeyError:
#                 grades[target] += sum(MODEL[target][col].values()) / len(MODEL[target][col])
#     for key in TARGET_SIZE:
#         grades[key] += TARGET_SIZE[key]
#     max_targ = max(grades, key=grades.get)
#     return {"result" : int(max_targ)}
#


