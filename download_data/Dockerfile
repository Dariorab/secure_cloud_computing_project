FROM python:3.10-slim
WORKDIR /pipeline
COPY requirements_download_data.txt /pipeline
RUN pip install -r requirements_download_data.txt
COPY download_data.py /pipeline
#COPY ETH_USD.csv /pipeline