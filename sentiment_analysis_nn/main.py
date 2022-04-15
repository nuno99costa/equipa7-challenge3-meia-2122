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

from sentiment_analysis_nn.dnn import DNN

# Load positive and negative tweets
all_positive_tweets, all_negative_tweets = DNN.load_tweets_from_nltk()

train_pos  = all_positive_tweets[:4000] # generating training set for positive tweets
train_neg  = all_negative_tweets[:4000] # generating training set for negative tweets

# Combine training data into one set
train_dataset = train_pos + train_neg 

# Build the vocabulary 
vocab = DNN.build_vocabulary(dataset=train_dataset)

# Creation of the model
model = DNN.classifier()

#TODO Add method that saves the trained model in our computer
#PATH = MY_PATH #TODO, example '/home/.../equipa7-(...)/checkpoints/model.pkl.gz'
#weights, state = model.init_from_file(PATH)

#TODO read file/database with sentences
sentence = "HEY, iphone 13 is great :)"
DNN.predict(sentence=sentence, vocab=vocab, model=model)