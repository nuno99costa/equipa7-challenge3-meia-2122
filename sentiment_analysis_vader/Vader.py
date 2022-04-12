from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from essential_generators import DocumentGenerator
gen = DocumentGenerator()
sid_obj = SentimentIntensityAnalyzer()

data = []
for i in range(100):
    data.append(gen.sentence())

#calculate the negative, positive, neutral and compound scores, plus verbal evaluation
def sentiment_vader(sentence):
    sentence = sentence.rstrip()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    neutral = sentiment_dict['neu']
    positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']

    if sentiment_dict['compound'] >= 0.1 :
        overall_sentiment = "Positive"

    elif sentiment_dict['compound'] <= - 0.1 :
        overall_sentiment = "Negative"

    else :
        overall_sentiment = "Neutral"
        
    return "{:-<100} {}".format(sentence, str([negative, neutral, positive, compound, overall_sentiment]))


def sentiment_vader_scale_amazon(sentence):
    sentence = sentence.rstrip()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    sentiment_dict['compound'] = (sentiment_dict['compound'] + 1) * 2.5

    if sentiment_dict['compound'] >= 4.5:
        overall_sentiment = "5"

    elif sentiment_dict['compound'] >= 3.5 and sentiment_dict['compound'] < 4.5 :
        overall_sentiment = "4"

    elif sentiment_dict['compound'] <= 1 :
        overall_sentiment = "1"

    elif sentiment_dict['compound'] <=2.5 and sentiment_dict['compound'] > 1:
        overall_sentiment = "2"

    else :
        overall_sentiment = "3"
  
    return "{:-<100} {}".format(sentence, str([overall_sentiment]))


with open('Output_Normal.txt', 'w', encoding="utf-8") as f:
    for i in range(100):
        f.write(sentiment_vader(data[i]))
        f.write('\n')
    f.write(sentiment_vader("I'm dumb!")) 
    f.write('\n')
    f.write(sentiment_vader("I'm very dumb! 😂")) 
    f.write('\n')
    f.write(sentiment_vader("He’s a great DJ. I seen him on Facebook Live. 😃")) 

with open('Output_Amazon.txt', 'w', encoding="utf-8") as f:
    for i in range(100):
        f.write(sentiment_vader_scale_amazon(data[i]))
        f.write('\n')    
    f.write(sentiment_vader_scale_amazon("He’s a great DJ. I seen him on Facebook Live. 😃"))
