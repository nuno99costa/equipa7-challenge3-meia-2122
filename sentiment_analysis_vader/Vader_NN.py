from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sid_obj = SentimentIntensityAnalyzer()
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Read in data
colnames=['target', 'id', 'date', 'flag', 'user', 'text'] 
df=pd.read_csv("sentiment_analysis_vader/data/training.1600000.processed.noemoticon.csv", names=colnames, encoding='ISO-8859-1')

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
print(f'Recall: {recall_score(df.dropna()["target"].values, df.dropna()["vader"].values, average="binary", pos_label=0) * 100}')
print(f'Precision: {precision_score(df.dropna()["target"].values, df.dropna()["vader"].values, average="binary", pos_label=0) * 100}')
print(f'F1: {f1_score(df.dropna()["target"].values, df.dropna()["vader"].values, average="binary", pos_label=0) * 100}')
print(f'Acertos: {accuracy_score(df.dropna()["target"].values, df.dropna()["vader"].values, normalize=False)}')