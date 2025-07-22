import uvicorn
from fastapi import FastAPI
from services.manager.classifier_service import router

app = FastAPI()



if __name__ == '__main__':
    app.include_router(router)
    uvicorn.run(app)
