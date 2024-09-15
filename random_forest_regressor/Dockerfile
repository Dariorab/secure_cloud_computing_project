FROM python:3.10-slim
WORKDIR /pipeline
COPY requirements_random_forest_regressor.txt /pipeline
RUN pip install -r requirements_random_forest_regressor.txt
COPY random_forest_regressor.py /pipeline
