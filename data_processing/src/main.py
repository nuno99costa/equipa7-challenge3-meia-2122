import pymongo
import requests
import hashlib
import nltk
from nltk.corpus import stopwords
from regex import W
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI
import re
import pandas as pd

KEYWORD_CUT_OFF = 10
TYPO_CUT_OFF = 3

app = FastAPI()

# connect to the database
def load_db():
    client = pymongo.MongoClient('database',
                                 username='root',
                                 password='root')
    db = client.data
    return db


def populate_db(data, collection):
    for o in data:
        try:
            collection.insert_one({
                '_id': o['_id'],
                'post_id': o['post_id'],
                'text': o['text'],
                'features': o['features']
            })
        except pymongo.errors.DuplicateKeyError:
            print('Duplicate:', o['text'])


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

emojis_pattern = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)


def process_workflow():
    db = load_db()
    raw_data = db.RAW_DATA
    data = raw_data.find()

    # Stop words
    stop_words = stopwords.words('english')

    tokenizer = nltk.tokenize.TweetTokenizer(
        preserve_case=False, reduce_len=True, strip_handles=True, match_phone_numbers=False)
    # stemmer = nltk.stem.PorterStemmer()
    lemmer = nltk.stem.WordNetLemmatizer()

    texts = []

    preprocessed_data = []

    emojis_backlog = []

    for tweet in data:
        text = tweet['text']

        text.encode('unicode-escape')

        # remove stock market tickers like $GE
        text = re.sub(r'\$\w*', '', text)

        # remove old style retweet text "RT"
        text = re.sub(r'^RT[\s]+', '', text)

        # Remove links
        text = re.sub(r'http\S+', '', text)

        # Seperate sentences
        sentences = nltk.tokenize.sent_tokenize(text, language='english')

        sentences.reverse()

        for sentence in sentences:
            # Emojis
            emojis = emojis_pattern.findall(sentence)
            emojis.extend(emojis_backlog)
            emojis_backlog = []

            sentence = emojis_pattern.sub('', sentence)

            tokenized_sentence = []
            # Tokenize sentence
            tokenized_sentence = tokenizer.tokenize(sentence)

            # Remove #
            filtered_sentence = [w[1:] if w.startswith('#') else w for w in tokenized_sentence]

            # Remove stop words
            filtered_sentence = [w for w in filtered_sentence if not w in stop_words]

            # Remove numbers
            filtered_sentence = [w for w in filtered_sentence if not re.match(r'^([,.\d]*)([,.]?\d*)$', w)]

            # Remove punctuation
            filtered_sentence = [w for w in filtered_sentence if not re.match(r'[^\w\s]', w)]

            # Check is sentence is empty
            if not filtered_sentence:
                emojis_backlog = emojis
                continue

            # Stem words
            # stemmed_sentence = [stemmer.stem(token) for token in filtered_sentence]
            final_sentence = [lemmer.lemmatize(token) for token in filtered_sentence]

            final_sentence.extend(emojis)

            o = {'post_id': tweet['_id'], 'text': final_sentence, 'features': []}
            texts.append(sentence)

            preprocessed_data.append(o)

    vectorizer = TfidfVectorizer(min_df=TYPO_CUT_OFF, max_df=.9, ngram_range=(1,2))
    features = vectorizer.fit_transform(texts)

    df = pd.DataFrame(features.todense(),columns=vectorizer.get_feature_names())

    keywords = df.columns.to_list()[:KEYWORD_CUT_OFF]

    print('Keywords:', keywords)

    typos = [t for t in vectorizer.stop_words_ if len(t.split(' ')) == 1 ]

    print('Typos:', typos)

    processed_data = []

    for o in preprocessed_data:
        text = [t for t in o['text'] if t not in typos]
        if not text:
            continue

        found_keywords = [t for t in o['text'] if t in keywords]
        o['text'] = text
        o['features'] = list(dict.fromkeys(found_keywords))

        # Rewrite new string
        sentence = ' '.join(text)

        #print(sentence)

        # Generate unique hash
        id = hashlib.md5(sentence.encode('unicode-escape')).hexdigest()
        o['_id'] = id

        processed_data.append(o)

    populate_db(processed_data, db.PROCESSED_DATA)

def request_sentiment_analysis():
    r = requests.get("http://sentiment_analysis_nn_pretrained/process_data")
    print(r)


@app.get("/process_data")
async def process_data():
    process_workflow()
    request_sentiment_analysis()
    return {'message': 'Operation Successful'}