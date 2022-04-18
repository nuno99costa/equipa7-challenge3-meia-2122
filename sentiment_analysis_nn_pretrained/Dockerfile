FROM python:3.9-buster

WORKDIR /code

COPY . .


RUN pip install textblob && \
    pip install pandas &&\
    pip install pymongo &&\
    pip install fastapi[all]

CMD [ "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]