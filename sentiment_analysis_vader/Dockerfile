FROM python:3.9-buster

WORKDIR /code

COPY . .

RUN pip install vaderSentiment && \
    pip install pandas && \
    pip install sklearn

CMD [ "python", "Vader_NN.py" ]