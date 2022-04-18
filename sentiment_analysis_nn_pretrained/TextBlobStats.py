from textblob import TextBlob
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from array import array
import pandas as pd
colnames=['target', 'id', 'date', 'flag', 'user', 'text'] 
df = pd.read_csv('C:\\Users\\diogo\\Downloads\\archive (2)\\training.1600000.processed.noemoticon.csv', names=colnames, encoding='latin-1', lineterminator='\n',low_memory=False)
tbresult = [TextBlob(i).sentiment.polarity for i in df]
tbpred = [0 if n < 0 else 1 for n in tbresult]
conmat = array(confusion_matrix(y_validation, tbpred, labels=[1,0]))
confusion = pd.DataFrame(conmat, index=['positive', 'negative'],
                         columns=['predicted_positive','predicted_negative'])
print ("Accuracy Score: {0:.2f}%".format(accuracy_score(y_validation, tbpred)*100))
print ("-"*80)
print ("Confusion Matrix\n")
print (confusion)
print ("-"*80)
print ("Classification Report\n")
print (classification_report(y_validation, tbpred))