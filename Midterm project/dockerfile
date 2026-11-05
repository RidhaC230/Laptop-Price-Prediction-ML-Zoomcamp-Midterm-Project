FROM python:3.13-slim

RUN pip install pipenv

WORKDIR /app

COPY Pipfile Pipfile.lock ./

RUN pipenv install --deploy --system

COPY predict.py model.bin ./


EXPOSE 5000


ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:5000", "predict:app"]