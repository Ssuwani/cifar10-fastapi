FROM python:3.8-slim

ENV PROJECT_DIR cifar10_classifier
WORKDIR /${PROJECT_DIR}
ADD ./requirements.txt /${PROJECT_DIR}/
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app/ /${PROJECT_DIR}/app/
COPY ./model/ /${PROJECT_DIR}/model/

WORKDIR /${PROJECT_DIR}/app/
CMD ["python app.py"]