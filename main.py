import uvicorn
from fastapi import FastAPI
from services.manager.classifier_service import router

app = FastAPI()
app.include_router(router)


if __name__ == '__main__':
    uvicorn.run(app="main:app",host="0.0.0.0", port=5000, reload=True)
