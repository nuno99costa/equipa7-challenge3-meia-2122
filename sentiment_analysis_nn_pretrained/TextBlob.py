from textblob import TextBlob
from essential_generators import DocumentGenerator
from tabulate import tabulate
gen = DocumentGenerator()
data = []
lista = []
col_names = ["Sentence", "Polarity (-1/1)","Subjectivity(0/1)"]
for i in range(100):
    Text = gen.sentence()
    lista.append(gen.sentence())
    def CalculatePolarity(Text):
        res=TextBlob(Text)
        return(res.sentiment.polarity)
    def CalculateSubjectivity(Text):
        res=TextBlob(Text)
        return(res.sentiment.subjectivity)
    lista.append(CalculatePolarity(Text))
    lista.append(CalculateSubjectivity(Text))
    data.append(lista)
    lista = []

print(tabulate(data, headers=col_names))
#*print(CalculatePolarity('This movie is amazing'))
#print(CalculateSubjectivity('This movie is amazing'))
#print(CalculatePolarity('This movie is Bad'))
#print(CalculateSubjectivity('This movie is Bad'))
#print(CalculatePolarity('This movie is normal'))
#print(CalculateSubjectivity('This movie is normal'))
#print(CalculatePolarity('I am not sure, maybe is goods or maybe is bad'))  