FROM python:3.13

WORKDIR /NaiveBayesClassifier

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python","main.py"]