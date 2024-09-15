FROM python:3.10-slim
WORKDIR /pipeline
COPY requirements_lstm_load_data.txt /pipeline
RUN pip install -r requirements_lstm_load_data.txt
COPY lstm_load_data.py /pipeline
COPY ETH_USD.csv /pipeline