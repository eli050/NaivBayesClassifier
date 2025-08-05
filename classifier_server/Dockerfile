FROM python:3.13.0-slim
WORKDIR /code
COPY classifier_server/requirements.txt .
RUN pip install -r requirements.txt
COPY /classifier_server .
EXPOSE 5002
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","5002"]