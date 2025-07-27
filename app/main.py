import uvicorn
from fastapi import FastAPI
from classifier_service import lifespan,router_app



app = FastAPI(lifespan=lifespan)
app.include_router(router_app)


if __name__ == '__main__':
    uvicorn.run(app="main:app",host="0.0.0.0", port=5002)