FROM python:3.10-slim
WORKDIR /pipeline
COPY requirements_lstm.txt /pipeline
RUN pip install -r requirements_lstm.txt
COPY lstm.py /pipeline