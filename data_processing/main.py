import pandas as pd
import hashlib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re

word_pattern = re.compile('^[a-z]')

def is_word(s):
    return word_pattern.match(s)

with open('Emoticon_Dict.p', 'rb') as fp:
    Emoticon_Dict = pickle.load(fp)

emoticon_pattern = re.compile(
    u'(' + u'|'.join(k for k in Emoticon_Dict) + u')')

# Stop words
nltk.download('stopwords')
stop_words = stopwords.words('english')

# texts

text = "Great movie. =P Would love to watch spiderman again! :))) 😀 \U0001F600"
text.encode('unicode-escape')
#text = text.casefold()

# Tokenize texts
#tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokenizer = nltk.tokenize.TweetTokenizer(
    preserve_case=False, reduce_len=True, strip_handles=True, match_phone_numbers=False)
stemmer = nltk.stem.PorterStemmer()
#lemmer = nltk.stem.WordNetLemmatizer()

sentences = nltk.tokenize.sent_tokenize(text, language='english')

tokenized_sentences = [tokenizer.tokenize(t) for t in sentences]
tokenized_sentences.reverse()
for i, sentence in enumerate(tokenized_sentences):

    # Checks if it is last sentence
    if i < len(tokenized_sentences)-1:
        j = 0
        # Adds emojis/emoticons to previous sentence
        for token in sentence:
            if is_word(token):
                break
            else:
                j += 1

        tokenized_sentences[i+1].extend(sentence[0:j])
        sentence = sentence[j:]

    #Check is sentence is empty
    if not sentence:
        continue

    # Remove stop words
    filtered_sentence = [w for w in sentence if not w in stop_words]

    # Stem words
    stemmed_sentence = [stemmer.stem(token) for token in filtered_sentence]

    # Rewrite new string
    sentence = ' '.join(stemmed_sentence)

    # Generate unique hash
    id = int(hashlib.md5(sentence.encode('utf-8')).hexdigest(), 16)

    print(sentence)

#tdidf = TfidfVectorizer(min_df=2, max_df=.9, ngram_range=(1, 2))
#features = tdidf.fit_transform(texts)
