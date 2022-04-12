from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import * 
from textblob import TextBlob 
from nltk import tokenize 
import pandas as pd
colnames=['target', 'id', 'date', 'flag', 'user', 'text'] 
df = pd.read_csv('C:\\Users\\diogo\\Downloads\\archive (2)\\training.1600000.processed.noemoticon.csv', names=colnames, encoding='latin-1', lineterminator='\n',low_memory=False)
df.drop_duplicates(subset ='text', keep = 'first', inplace = True)
df['text'] = df['text'].astype('str')
def get_polarity(text):
    return TextBlob(text).sentiment.polarity
df['Polarity'] = df['text'].apply(get_polarity)
df['Sentiment_Type']=''
df.loc[df.Polarity>0,'Sentiment_Type']='POSITIVE'
df.loc[df.Polarity==0,'Sentiment_Type']='NEUTRAL'
df.loc[df.Polarity<0,'Sentiment_Type']='NEGATIVE'

print(df.Sentiment_Type.value_counts())
print(df.Sentiment_Type.value_counts(normalize=True))




