FROM python:3.10

ADD . /patient_survival_predictor
WORKDIR /patient_survival_predictor

RUN pip install --no-cache-dir -r ./requirements/requirements.txt

COPY . .


EXPOSE 8001

CMD ["python","./model/trained_models/model.py"]

CMD ["python", "./predict.py"]
