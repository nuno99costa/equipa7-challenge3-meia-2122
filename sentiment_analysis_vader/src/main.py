from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fastapi import FastAPI
from pymongo import MongoClient
import requests
import os
import json

sid_obj = SentimentIntensityAnalyzer()
app = FastAPI()


# connect to the database
def load_db():
    client = MongoClient("database",
                         username='root',
                         password='root')
    db = client.data
    return db


def calculate_polarity(text):
    res =  sid_obj.polarity_scores(text)
    res = (((res +1) * (5 - 1)) / (1 + 1)) + 1


def save_result(results, tweet, sentiment):
    results.update_one(
        {'_id': tweet["_id"]},
        {"$set": {'sentiment_vader': sentiment}},
        upsert=True
    )


@app.get("/process_data")
async def process_data():
    database = load_db()
    data = database.PROCESSED_DATA
    results = database.RESULTS
    for tweet in data.find():
        text = " ".join(tweet['text'])
        print(text)
        sentiment = calculate_polarity(text)
        save_result(results, tweet, sentiment)
    return {'message': 'Operation Successful'}
