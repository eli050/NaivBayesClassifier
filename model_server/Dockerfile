FROM python:3.13.0-slim
WORKDIR /code
COPY model_server/requirements.txt .
RUN pip install -r requirements.txt
COPY /app /code/app
COPY /Data /code/Data
COPY /model_server /code/model_server
EXPOSE 5001
CMD ["uvicorn", "model_server.main:app", "--host", "0.0.0.0", "--port", "5001"]