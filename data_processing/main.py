from pymongo import MongoClient
import hashlib
import nltk
from nltk.corpus import stopwords
from regex import W
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd

# connect to the database
def load_db():
    client = MongoClient('database',
                         username='root',
                         password='root')
    db = client.data
    return db

def populate_db(data, collection):
    for o in data:
        collection.insert_one({
            '_id': o['_id'],
            'post_id': o['post_id'],
            'text': o['text'],
            'features': o['features']
        })

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def main():
    db = load_db()
    raw_data = db.RAW_DATA
    data =  raw_data.find()

    # Stop words
    stop_words = stopwords.words('english')

    tokenizer = nltk.tokenize.TweetTokenizer(
        preserve_case=False, reduce_len=True, strip_handles=True, match_phone_numbers=False)
    #stemmer = nltk.stem.PorterStemmer()
    lemmer = nltk.stem.WordNetLemmatizer()

    """ texts = []
    for tweet in data:
        texts.append(tweet['text']) """

    processed_data = []
    
    for tweet in data:
        text = tweet['text']

        # remove stock market tickers like $GE
        text = re.sub(r'\$\w*', '', text)

        # remove old style retweet text "RT"
        text = re.sub(r'^RT[\s]+', '', text)

        #Remove links
        text = re.sub(r'http\S+', '', text)

        #Seperate sentences
        sentences = nltk.tokenize.sent_tokenize(text, language='english')
        
        for sentence in sentences:
            
            tokenized_sentence = []
            #Tokenize sentence            
            tokenized_sentence = tokenizer.tokenize(sentence)

            #Remove #
            filtered_sentence = [w[1:] if w.startswith('#') else w for w in tokenized_sentence ]

            # Remove stop words
            filtered_sentence = [w for w in filtered_sentence if not w in stop_words]

            # Remove numbers
            filtered_sentence = [w for w in filtered_sentence if not re.match(r'^([,.\d]*)([,.]?\d*)$', w)]

            # Remove punctuation
            filtered_sentence = [w for w in filtered_sentence if not re.match(r'[^\w\s]', w)]

            #Check is sentence is empty
            if not filtered_sentence:
                continue

            # Stem words
            #stemmed_sentence = [stemmer.stem(token) for token in filtered_sentence]
            final_sentence = [lemmer.lemmatize(token) for token in filtered_sentence]

            # Rewrite new string
            sentence = ' '.join(final_sentence)

            # Generate unique hash
            id = hashlib.md5(sentence.encode('unicode-escape')).hexdigest()

            print(sentence)

            o = {'post_id':tweet['_id'], '_id': id, 'text': final_sentence, 'features': []}
            #texts.append(sentence)
            
            #print('\t',o)
            
            processed_data.append(o)
    
    populate_db(processed_data, db.PROCESSED_DATA)

    """ vectorizer = TfidfVectorizer(min_df=4, max_df=.7, ngram_range=(1,2))

    features = vectorizer.fit_transform(texts)

    print(vectorizer.get_stop_words())

    df = pd.DataFrame(features.todense(), columns=vectorizer.get_feature_names_out()) """
    #df.to_csv('a.csv')

if __name__ == "__main__":
    main()