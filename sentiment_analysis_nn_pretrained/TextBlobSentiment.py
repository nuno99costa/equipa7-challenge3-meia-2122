from nltk.sentiment.util import * 
from textblob import TextBlob 
from nltk import tokenize 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
colnames=['target', 'id', 'date', 'flag', 'user', 'text'] 
df = pd.read_csv('data/training.1600000.processed.noemoticon.csv', names=colnames, encoding='ISO-8859-1', lineterminator='\n',low_memory=False)
df.drop_duplicates(subset ='text', keep = 'first', inplace = True)
df['text'] = df['text'].astype('str')
def get_polarity(text):
    return TextBlob(text).sentiment.polarity
def get_polarity_Int(text):
    textToF=TextBlob(text).sentiment.polarity
    if textToF >= 0: 
        return 4
    else: 
        return 0

df['PolarityMetrics']=df.apply(lambda x: get_polarity_Int(x['text']), axis=1)   
df['Polarity'] = df['text'].apply(get_polarity)
df['Sentiment_Type']=''
df.loc[df.Polarity>0,'Sentiment_Type']='POSITIVE'   
df.loc[df.Polarity==0,'Sentiment_Type']='NEUTRAL'
df.loc[df.Polarity<0,'Sentiment_Type']='NEGATIVE'


print(f'Accuracy: {accuracy_score(df.target.to_list(), df.dropna()["PolarityMetrics"].values) * 100}')
print(f'Recall: {recall_score(df.target.to_list(), df.dropna()["PolarityMetrics"].values, average="binary", pos_label=0) * 100}')
print(f'Precision: {precision_score(df.target.to_list(), df.dropna()["PolarityMetrics"].values, average="binary", pos_label=0) * 100}')
print(f'F1: {f1_score(df.target.to_list(), df.dropna()["PolarityMetrics"].values, average="binary", pos_label=0) * 100}')
print(f'Acertos: {accuracy_score(df.target.to_list(), df.dropna()["PolarityMetrics"].values, normalize=False)}')
#print(df.Sentiment_Type.value_counts())
#print(df.Sentiment_Type.value_counts(normalize=True))




