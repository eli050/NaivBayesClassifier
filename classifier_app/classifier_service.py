import requests
from fastapi import FastAPI, APIRouter, Request
from contextlib import asynccontextmanager


BASE_URL = "http://model_host:5001/model"


router_app = APIRouter()

ml_models = dict()




@asynccontextmanager
async def lifespan(app:FastAPI):
    """
       Lifespan event for the FastAPI application.
       Tries to fetch the model from the external service at BASE_URL
       and updates the global `ml_models` dictionary with the response.
       """
    try:
        response = requests.get(BASE_URL)
        ml_models.update(response.json())
    except Exception as e:
        print(e)
        raise
    yield


@router_app.get("/predict")
def get_grade(request: Request):
    """
        Endpoint that receives passenger-related parameters (like in the Titanic dataset)
        and returns the predicted target class based on the loaded model.

        Process:
        1. Builds a dictionary from the input parameters.
        2. For each possible target, calculates a score by summing the model's weights
           for each feature. If a value is missing in the model, it uses the average.
        3. Adds an extra score from `target_size` to each target.
        4. Returns the target with the highest total score.
        """
    grades = dict()
    dict_data = dict(request.query_params)
    for target in ml_models["model"]:
        grades[target] = 0
        for col, value in dict_data.items():
            try:
                grades[target] += ml_models["model"][target][col][value]
            except KeyError:
                grades[target] += -7
    for key in ml_models["target_size"]:
        grades[key] += ml_models["target_size"][key]
    max_targ = max(grades, key=grades.get)
    return {"result" : str(max_targ)}
