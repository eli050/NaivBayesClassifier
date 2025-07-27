import uvicorn
from fastapi import FastAPI
from services.manager.model_service import router,lifespan



app = FastAPI(lifespan=lifespan)
app.include_router(router)


if __name__ == '__main__':
    uvicorn.run(app="main:app",host="0.0.0.0", port=5001)
