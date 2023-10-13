FROM svizor/zoomcamp-model:3.10.12-slim

WORKDIR /app

COPY Pipfile Pipfile.lock ./

RUN pip install pipenv

RUN pipenv install --system --deploy

COPY predict.py ./

EXPOSE 5001

CMD [ "flask", "--app", "predict", "run" ]