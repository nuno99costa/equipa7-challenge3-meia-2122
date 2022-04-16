import os 
import random as rnd
import re
import string
from process_text import process_text
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# import relevant libraries
install("jaxlib")
install("trax")
import trax

# import trax.fastmath.numpy
import trax.fastmath.numpy as np # the same a Jax
from trax import fastmath

# import trax.layers
from trax import layers as tl

import nltk
from nltk.corpus import stopwords, twitter_samples 
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer


class DNN:

    def __init__(self):
        nltk.download('twitter_samples')
        nltk.download('stopwords')

        self.stopwords_english = stopwords.words('english')
        self.stemmer = PorterStemmer()

        
    def load_tweets_from_nltk():
        '''
        Load tweets from Natural Language Toolkit (NLTK), so we can train our model
        '''
        all_positive_tweets = twitter_samples.strings('positive_tweets.json')
        all_negative_tweets = twitter_samples.strings('negative_tweets.json')  
        return all_positive_tweets, all_negative_tweets

    def process_tweet(self, tweet):
        return process_text(tweet)


    def build_vocabulary(self, train_x):
        vocab = {'__PAD__': 0, '__</e>__': 1, '__UNK__': 2} 

        # Create vocabulary using words from training data
        for tweet in train_x: 
            for sentence in self.process_tweet(tweet):
                for word in sentence:
                    if word not in vocab: 
                        vocab[word] = len(vocab)
        
        return vocab


    def classifier(vocab_size=10000, embedding_dim=256, output_dim=2, mode='train'):
        '''
        Method that creates the Model
        '''
        # create embedding layer
        embed_layer = tl.Embedding(
            vocab_size=vocab_size, # Size of the vocabulary
            d_feature=embedding_dim)  # Embedding dimension
        
        # Create a mean layer, to create an "average" word embedding
        mean_layer = tl.Mean(axis=1)
        
        # Create a dense layer, one unit for each output
        dense_output_layer = tl.Dense(n_units = output_dim)

        # Create the log softmax layer (no parameters needed)
        log_softmax_layer = tl.LogSoftmax()
        
        # Use tl.Serial combinator
        model = tl.Serial(
        embed_layer, # embedding layer
        mean_layer, # mean layer
        dense_output_layer, # dense output layer 
        log_softmax_layer # log softmax layer
        )
        
        # return the model of type
        return model


    def tweet_to_tensor(self, tweet, vocab_dict, unk_token='__UNK__', verbose=False):
        '''
        Input: 
            tweet - A string containing a tweet
            vocab_dict - The words dictionary
            unk_token - The special string for unknown tokens
            verbose - Print info durign runtime
        Output:
            tensor_l - A python list with
            
        '''  
        
        ### START CODE HERE (Replace instances of 'None' with your code) ###
        # Process the tweet into a list of words
        # where only important words are kept (stop words removed)
        word_l = self.process_tweet(tweet)
        
        if verbose:
            print("List of words from the processed tweet:")
            print(word_l)
            
        # Initialize the list that will contain the unique integer IDs of each word
        tensor_l = []
        
        # Get the unique integer ID of the __UNK__ token
        unk_ID = vocab_dict[unk_token]
        
        if verbose:
            print(f"The unique integer ID for the unk_token is {unk_ID}")
            
        # for each word in the list:
        for word in word_l:
            
            # Get the unique integer ID.
            # If the word doesn't exist in the vocab dictionary,
            # use the unique ID for __UNK__ instead.
            word_ID = vocab_dict[word] if word in vocab_dict else unk_ID
        ### END CODE HERE ###
            
            # Append the unique integer ID to the tensor list.
            tensor_l.append(word_ID) 
        
        return tensor_l



    def predict(self, sentence, vocab, model):
        inputs = np.array(self.tweet_to_tensor(sentence, vocab_dict=vocab))
        
        # Batch size 1, add dimension for batch, to work with the model
        inputs = inputs[None, :]  
        
        # predict with the model
        preds_probs = model(inputs) # log softmax result
        
        # Turn probabilities into categories
        preds = int(preds_probs[0, 1] > preds_probs[0, 0])
        
        sentiment = "negative"
        if preds == 1:
            sentiment = 'positive'

        return preds, sentiment

