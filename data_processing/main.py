from string import punctuation
from pymongo import MongoClient
import hashlib
import nltk
from nltk.corpus import stopwords
#from sklearn.feature_extraction.text import TfidfVectorizer
import re

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

def main():
    # Stop words
    stop_words = stopwords.words('english')

    db = load_db()
    raw_data = db.RAW_DATA

    # Tokenize texts
    tokenizer = nltk.tokenize.TweetTokenizer(
        preserve_case=False, reduce_len=True, strip_handles=True, match_phone_numbers=False)
    stemmer = nltk.stem.PorterStemmer()
    #lemmer = nltk.stem.WordNetLemmatizer()


    processed_data = []
    for tweet in raw_data.find():
        text = tweet['text']

        #Remove links
        text = re.sub(r'http\S+', '', text)

        #Seperate sentences
        sentences = nltk.tokenize.sent_tokenize(text, language='english')

        for sentence in sentences:
            print('NEW SENTENCE:\n\t', sentence)

            #Tokenize sentence            
            tokenized_sentence = tokenizer.tokenize(sentence)

            #Remove # 
            for token in tokenized_sentence:
                if token.startswith('#'):
                    token = token[1:]
                    
            filtered_sentence = [w[1:] for w in tokenized_sentence if token.startswith('#') else w]

            # Remove stop words
            filtered_sentence = [w for w in tokenized_sentence if not w in stop_words]

            filtered_sentence = [w for w in tokenized_sentence if not w in punctuation]

            #Check is sentence is empty
            if not filtered_sentence:
                continue

            # Stem words
            stemmed_sentence = [stemmer.stem(token) for token in filtered_sentence]

            # Rewrite new string
            sentence = ' '.join(stemmed_sentence)

            # Generate unique hash
            id = hashlib.md5(sentence.encode('unicode-escape')).hexdigest()

            o = {'post_id':tweet['_id'], '_id': id, 'text': stemmed_sentence, 'features': []}
            print('\t',o)
            processed_data.append(o)

    populate_db(processed_data, db.PROCESSED_DATA)

    #tdidf = TfidfVectorizer(min_df=2, max_df=.9, ngram_range=(1, 2))
    #features = tdidf.fit_transform(texts)

if __name__ == "__main__":
    main()
