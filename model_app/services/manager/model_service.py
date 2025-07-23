from fastapi import APIRouter,FastAPI
from contextlib import asynccontextmanager
from services.manager.manag_app import ManagementApp

TRAIN_PATH = "././Data/train.csv"
TEST_PATH = "././Data/test.csv"
router = APIRouter()



ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models.update(ManagementApp.start_flow())


@router.get("/model")
def get_model():
    print(ml_models["grade"])
    return ml_models["model"], ml_models["target_size"]




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


