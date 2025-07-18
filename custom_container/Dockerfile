FROM python:3.7-slim 

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y build-essential &&  pip install --upgrade pip

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY inference.py /opt/program/inference.py 

ENV SAGEMAKER_PROGRAM inference.py 

WORKDIR /opt/program