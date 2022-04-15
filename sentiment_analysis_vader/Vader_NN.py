from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sid_obj = SentimentIntensityAnalyzer()
import pandas as pd
from sklearn.metrics import accuracy_score

# Read in data
colnames=['target', 'id', 'date', 'flag', 'user', 'text'] 
#colnames1=['id', 'entity', 'sentiment', 'text', "vader"] 
df=pd.read_csv('data/training.1600000.processed.noemoticon.csv', names=colnames, encoding='ISO-8859-1')
#df1=pd.read_csv('data/training.1600000.processed.noemoticon.csv', names=colnames1, encoding='ISO-8859-1', lineterminator='\n')

# Calculate polarity score using Vader
def get_vader_score(sentence): 
    compound=sid_obj.polarity_scores(sentence)['compound']
    if compound >= 0: 
        return 4
    #elif (compound >= -0.1) and (compound <=0.1): 
    #    return 2
    else: 
        return 0

def get_vader_score_text(sentence): 
    compound=sid_obj.polarity_scores(sentence)['compound']
    if compound > 0.05: 
        return "Positive"
    elif (compound >= -0.05) and (compound <=0.05): 
        return "Neutral"
    else: 
        return "Negative"

df['vader']=df.apply(lambda x: get_vader_score(x['text']), axis=1)
#df1['vader']=df.apply(lambda x: get_vader_score_text(x['text']), axis=1)

# Evaluate results
print(f'Accuracy: {accuracy_score(df.dropna()["target"].values, df.dropna()["vader"].values) * 100}')
print(f'Acertos: {accuracy_score(df.dropna()["target"].values, df.dropna()["vader"].values, normalize=False)}')
#print(f'Accuracy: {accuracy_score(df1.dropna()["sentiment"].values, df1.dropna()["vader"].values)}')