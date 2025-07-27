FROM python:3.13

WORKDIR /classifier_app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/classifier_app

CMD ["python","main.py"]