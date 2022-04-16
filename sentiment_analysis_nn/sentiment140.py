'''
How to run:
    In the terminal, go to this folder (sentiment_analysis_nn), make sure your docker desktop is running and:
        docker build --tag custom_dnn .
        docker run custom_dnn

Missing Steps:
    - Terminar de testar a Deep Neural Network
    - Modificar o build vocabulary para ligar ao data processing em vez de utilizar o metodo "process_tweet"
    - Guardar o modelo localmente para nao termos de estar sempre a treinar
    - Analise de performance deste modelo vs outros modelos (vader e text_blob)
'''

from dnn import DNN
import pandas as pd
from process_text import process_tweet

colnames=['target', 'id', 'date', 'flag', 'user', 'text'] 
df = pd.read_csv('./training.1600000.processed.noemoticon.csv', names=colnames, encoding='latin-1', lineterminator='\n',low_memory=False)
df.drop_duplicates(subset ='text', keep = 'first', inplace = True)

texts = df.text.to_list()
targets = df.target.to_list()

processed_texts = []
for text, target in texts, targets:
    processed_texts.extend(process_tweet(text))

l = len(processed_texts)*.8

train_dataset = processed_texts[:l]
test_dataset = processed_texts[l:]

# Build the vocabulary 
vocab = DNN.build_vocabulary(dataset=train_dataset)

# Creation of the model
model = DNN.classifier()

#TODO Add method that saves the trained model in our computer
#PATH = MY_PATH #TODO, example '/home/.../equipa7-(...)/checkpoints/model.pkl.gz'
#weights, state = model.init_from_file(PATH)

#TODO read file/database with sentences
sentence = "HEY, iphone 13 is great :)"
#DNN.predict(sentence=sentence, vocab=vocab, model=model)