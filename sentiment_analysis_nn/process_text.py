import nltk
from nltk.corpus import stopwords
import re

# Stop words
stop_words = stopwords.words('english')

tokenizer = nltk.tokenize.TweetTokenizer(
    preserve_case=False, reduce_len=True, strip_handles=True, match_phone_numbers=False)
# stemmer = nltk.stem.PorterStemmer()
lemmer = nltk.stem.WordNetLemmatizer()

emojis_pattern = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)

def process_text(text):

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
        sentence = emojis_pattern.sub('', sentence)

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

        # Stem words
        final_sentence = [lemmer.lemmatize(token) for token in filtered_sentence]

        final_sentence.extend(emojis)
        
        yield final_sentence

