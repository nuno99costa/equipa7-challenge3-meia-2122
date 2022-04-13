FROM python:3.9-buster

WORKDIR /code

COPY . .


RUN pip install textblob && \
    pip install pandas

CMD [ "python", "TextBlobSentiment.py" ]