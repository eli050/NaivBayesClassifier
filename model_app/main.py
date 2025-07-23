# import sys
# from pathlib import Path
# project_root = Path(__file__).resolve().parent
# sys.path.insert(0,str(project_root))
import uvicorn
from fastapi import FastAPI
from services.manager.model_service import router,lifespan



app = FastAPI(lifespan=lifespan)
app.include_router(router)


if __name__ == '__main__':
    uvicorn.run(app="main:app",host="0.0.0.0", port=5000, reload=True)
