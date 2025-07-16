import uvicorn
from fastapi import FastAPI
from services.manag_ui.server_ui import router

app = FastAPI()



if __name__ == '__main__':
    app.include_router(router)
    uvicorn.run(app)
