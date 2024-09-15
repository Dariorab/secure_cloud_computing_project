FROM python:3.10-slim
WORKDIR /pipeline
COPY requirements_load_data.txt /pipeline
RUN pip install -r requirements_load_data.txt
COPY load_data.py /pipeline
COPY ETH_USD.csv /pipeline