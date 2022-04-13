from pymongo import MongoClient
import requests
import os
import json

bearer_token = os.environ.get("BEARER_TOKEN")
search_url = "https://api.twitter.com/2/tweets/search/recent"
query_params = {'query': '(-is:retweet lang:en is:verified) ("iPhone 13 Pro" OR #iPhone13Pro)', 'max_results': 100, 'tweet.fields': 'created_at'}


# connect to the database
def load_db():
    client = MongoClient("database",
                         username='root',
                         password='root')
    db = client.data
    return db.RAW_DATA


def bearer_oauth(r):
    auth = f"Bearer {bearer_token}"
    auth = auth.replace('"', '')
    r.headers["Authorization"] = auth
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r


def connect_to_endpoint(url, params):
    response = requests.get(url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def populate_db(json_response, raw_data_collection):
    for tweet in json_response["data"]:
        print(tweet)
        raw_data_collection.insert_one({
            '_id': tweet["id"],
            'date': tweet["created_at"],
            'text': tweet["text"]
        })


def main():
    raw_data_collection = load_db()
    json_response = connect_to_endpoint(search_url, query_params)
    populate_db(json_response, raw_data_collection)
    print(json.dumps(json_response, indent=4, sort_keys=True))


if __name__ == "__main__":
    main()
