from fastapi import APIRouter,FastAPI
from contextlib import asynccontextmanager
from .pipeline import ManagementApp

router = APIRouter()



ml_models = dict()

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
        Lifespan event for the FastAPI application.
        Initializes the machine learning model by running the full pipeline
        via ManagementApp. If an error occurs during the process, it is stored
        in the ml_models dictionary under the key 'error'.
    """
    try:
        ml_models.update(ManagementApp.start_flow())
    except Exception as e:
        ml_models["error"] = e
    yield



@router.get("/model")
def get_model():
    """
       Endpoint that returns the trained model and its target size weights.
       If the model has not been successfully trained, returns an error message.
    """
    if "error" not in ml_models:
        print(ml_models["grade"])
        return {"model":ml_models["model"], "target_size":ml_models["target_size"]}
    else:
        return {"error":"The model is not yet trained."}







