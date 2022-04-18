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
from trax.supervised import training

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

        
    def load_tweets_from_nltk(self):
        '''
        Load tweets from Natural Language Toolkit (NLTK), so we can train our model
        '''
        all_positive_tweets = twitter_samples.strings('positive_tweets.json')
        all_negative_tweets = twitter_samples.strings('negative_tweets.json')  
        return all_positive_tweets, all_negative_tweets

    def process_tweet(self, tweet):
        return process_text(tweet)


    def build_vocabulary(self, dataset):
        vocab = {'__PAD__': 0, '__</e>__': 1, '__UNK__': 2} 

        # Create vocabulary using words from training data
        for tweet in dataset: 
            for word in self.process_tweet(tweet):
                if word not in vocab: 
                    vocab[word] = len(vocab)
        
        return vocab


    def classifier(self, vocab_size=10000, embedding_dim=256, output_dim=2, mode='train'):
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
        Transforms words into IDs from the vocabulary, so our NN won't have words but IDs
        Input: 
            tweet - A string containing a tweet
            vocab_dict - The words dictionary
            unk_token - The special string for unknown tokens
            verbose - Print info durign runtime
        Output:
            tensor_l - A python list with
            
        '''  
        
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

    # Create the training data generator
    def train_generator(self, batch_size, train_pos, train_neg, vocab, shuffle = False):
        return self.data_generator(data_pos=train_pos, data_neg=train_neg, batch_size=batch_size, loop=True, vocab_dict=vocab, shuffle=shuffle)


    # Create the validation data generator
    def val_generator(self, batch_size, val_pos, val_neg, vocab, shuffle = False):
        return self.data_generator(data_pos=val_pos, data_neg=val_neg,  batch_size=batch_size, loop=True, vocab_dict=vocab, shuffle=shuffle)


    # Create the validation data generator
    def test_generator(self, batch_size, val_pos, val_neg, vocab, shuffle = False):
        return self.data_generator(data_pos=val_pos, data_neg=val_neg, batch_size=batch_size, loop=False, vocab_dict=vocab, shuffle=shuffle)


    def data_generator(self, data_pos, data_neg, batch_size, loop, vocab_dict, shuffle=False):
        '''
        
        Method that generates batches of data to be used in training/evaluation of models

        Input: 
            data_pos - Set of positive examples
            data_neg - Set of negative examples
            batch_size - number of samples per batch. Must be even
            loop - True or False
            vocab_dict - The words dictionary
            shuffle - Shuffle the data order
        Yield:
            inputs - Subset of positive and negative examples
            targets - The corresponding labels for the subset
            example_weights - An array specifying the importance of each example
            
        '''     
        # make sure the batch size is an even number
        # to allow an equal number of positive and negative samples
        assert batch_size % 2 == 0
        
        # Number of positive examples in each batch is half of the batch size
        # same with number of negative examples in each batch
        n_to_take = batch_size // 2
        
        # Use pos_index to walk through the data_pos array
        # same with neg_index and data_neg
        pos_index = 0
        neg_index = 0
        
        len_data_pos = len(data_pos)
        len_data_neg = len(data_neg)
        
        # Get and array with the data indexes
        pos_index_lines = list(range(len_data_pos))
        neg_index_lines = list(range(len_data_neg))
        
        # shuffle lines if shuffle is set to True
        if shuffle:
            rnd.shuffle(pos_index_lines)
            rnd.shuffle(neg_index_lines)
            
        stop = False
        
        # Loop indefinitely
        while not stop:  
            
            # create a batch with positive and negative examples
            batch = []
            
            # First part: Pack n_to_take positive examples 
            # Start from pos_index and increment i up to n_to_take
            for i in range(n_to_take):
                        
                # If the positive index goes past the positive dataset lenght,
                if pos_index >= len_data_pos: 
                    
                    # If loop is set to False, break once we reach the end of the dataset
                    if not loop:
                        stop = True
                        break
                    
                    # If user wants to keep re-using the data, reset the index
                    pos_index = 0
                    
                    if shuffle:
                        # Shuffle the index of the positive sample
                        rnd.shuffle(pos_index_lines)
                        
                # get the tweet as pos_index
                tweet = data_pos[pos_index_lines[pos_index]]
                
                # convert the tweet into tensors of integers representing the processed words
                tensor = self.tweet_to_tensor(tweet, vocab_dict)
                
                # append the tensor to the batch list
                batch.append(tensor)
                
                # Increment pos_index by one
                pos_index = pos_index + 1

            # Second part: Pack n_to_take negative examples
        
            # Using the same batch list, start from neg_index and increment i up to n_to_take
            for i in range(n_to_take):
                # If the negative index goes past the negative dataset length,
                if neg_index >= len_data_neg:        
                    # If loop is set to False, break once we reach the end of the dataset
                    if not loop:
                        stop = True
                        break
                        
                    # If user wants to keep re-using the data, reset the index
                    neg_index = 0
                    
                    if shuffle:
                        # Shuffle the index of the negative sample
                        rnd.shuffle(neg_index_lines)
                # get the tweet as pos_index
                tweet = data_neg[neg_index_lines[neg_index]]
                
                # convert the tweet into tensors of integers representing the processed words
                tensor = self.tweet_to_tensor(tweet, vocab_dict)
                
                # append the tensor to the batch list
                batch.append(tensor)
                
                # Increment neg_index by one
                neg_index += 1

            if stop:
                break

            # Update the start index for positive data 
            # so that it's n_to_take positions after the current pos_index
            pos_index += n_to_take
            
            # Update the start index for negative data 
            # so that it's n_to_take positions after the current neg_index
            neg_index += n_to_take
            
            # Get the max tweet length (the length of the longest tweet) 
            # (you will pad all shorter tweets to have this length)
            max_len = max([len(t) for t in batch]) 
             
            # Initialize the input_l, which will 
            # store the padded versions of the tensors
            tensor_pad_l = []
            # Pad shorter tweets with zeros
            for tensor in batch:
                # Get the number of positions to pad for this tensor so that it will be max_len long
                n_pad = max_len - len(tensor)
                
                # Generate a list of zeros, with length n_pad
                pad_l = [0] * n_pad
                
                # concatenate the tensor and the list of padded zeros
                tensor_pad = tensor + pad_l
                
                # append the padded tensor to the list of padded tensors
                tensor_pad_l.append(tensor_pad)

            # convert the list of padded tensors to a numpy array
            # and store this as the model inputs
            inputs = np.array(tensor_pad_l)
    
            # Generate the list of targets for the positive examples (a list of ones)
            # The length is the number of positive examples in the batch
            target_pos = [1] * n_to_take
            
            # Generate the list of targets for the negative examples (a list of ones)
            # The length is the number of negative examples in the batch
            target_neg = [0] * n_to_take
            
            # Concatenate the positve and negative targets
            target_l = target_pos + target_neg
            
            # Convert the target list into a numpy array
            targets = np.array(target_l)

            # Example weights: Treat all examples equally importantly
            example_weights = np.ones_like(targets)
            
            # note we use yield and not return
            yield inputs, targets, example_weights


    
    def train_model(self, train_pos, train_neg, val_pos, val_neg, vocab, classifier, n_steps):
        '''
        Input: 
            classifier - the model you are building
            train_task - Training task
            eval_task - Evaluation task
            n_steps - the evaluation steps
            output_dir - folder to save your files
        Output:
            trainer -  trax trainer
        '''
        batch_size = 16
        rnd.seed(271)

        train_task = training.TrainTask(
            labeled_data=self.train_generator(batch_size=batch_size, train_pos=train_pos, train_neg=train_neg, vocab=vocab, shuffle=True),
            loss_layer=tl.CrossEntropyLoss(),
            optimizer=trax.optimizers.Adam(0.01),
            n_steps_per_checkpoint=10,
        )

        eval_task = training.EvalTask(
            labeled_data=self.val_generator(batch_size=batch_size, val_pos=val_pos, val_neg=val_neg, vocab=vocab, shuffle=True),
            metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
        )
        
        output_dir = '~/model/'

        training_loop = training.Loop(
                                classifier, # The learning model
                                train_task, # The training task
                                eval_tasks = [eval_task], # The evaluation task
                                output_dir = output_dir) # The output directory

        training_loop.run(n_steps = n_steps)

        # Return the training_loop, since it has the model.
        return training_loop


    def validate_predictions_trained_model(self, training_loop):
        '''
        Method used to visually verify if the predictions of the model are as expected
        '''
        tmp_data_gen = self.train_generator(batch_size = 4)

        # Call the data generator to get one batch and its targets
        tmp_inputs, tmp_targets, tmp_example_weights = next(tmp_data_gen)

        # feed the tweet tensors into the model to get a prediction
        tmp_pred = training_loop.eval_model(tmp_inputs)

        # turn probabilites into category predictions
        tmp_is_positive = tmp_pred[:,1] > tmp_pred[:,0]
        for i, p in enumerate(tmp_is_positive):
            print(f"Neg log prob {tmp_pred[i,0]:.4f}\tPos log prob {tmp_pred[i,1]:.4f}\t is positive? {p}\t actual {tmp_targets[i]}")


    def see_output_of_train_generator(self):
        # Get a batch from the train_generator and inspect.
        inputs, targets, example_weights = next(self.train_generator(4, shuffle=True))

        # this will print a list of 4 tensors padded with zeros
        print(f'Inputs: {inputs}')
        print(f'Targets: {targets}')
        print(f'Example Weights: {example_weights}')

        # Test the train_generator

        # Create a data generator for training data,
        # which produces batches of size 4 (for tensors and their respective targets)
        tmp_data_gen = self.train_generator(batch_size = 4)

        # Call the data generator to get one batch and its targets
        tmp_inputs, tmp_targets, tmp_example_weights = next(tmp_data_gen)

        print(f"The inputs shape is {tmp_inputs.shape}")
        print(f"The targets shape is {tmp_targets.shape}")
        print(f"The example weights shape is {tmp_example_weights.shape}")

        for i,t in enumerate(tmp_inputs):
            print(f"input tensor: {t}; target {tmp_targets[i]}; example weights {tmp_example_weights[i]}")

    
    def compute_accuracy(self, predictions, y, y_weights):
        """
        Input: 
            predictions: a tensor of shape (dim_batch, output_dim) 
            y: a tensor of shape (dim_batch,) with the true labels
            y_weights: a n.ndarray with the a weight for each example
        Output: 
            accuracy: a float between 0-1 
            weighted_num_correct (np.float32): Sum of the weighted correct predictions
            sum_weights (np.float32): Sum of the weights
        """
        # Create an array of booleans, 
        # True if the probability of positive sentiment is greater than
        # the probability of negative sentiment
        # else False
        is_pos =  predictions[:, 1] > predictions[:, 0] 

        # convert the array of booleans into an array of np.int32
        is_pos_int = is_pos.astype(np.int32)
        
        # compare the array of predictions (as int32) with the target (labels) of type int32
        correct = is_pos_int == y

        # Count the sum of the weights.
        sum_weights = np.sum(y_weights)
        
        # convert the array of correct predictions (boolean) into an arrayof np.float32
        correct_float = correct.astype(np.float32)
        
        # Multiply each prediction with its corresponding weight.
        weighted_correct_float = correct_float * y_weights

        # Sum up the weighted correct predictions (of type np.float32), to go in the
        # denominator.
        weighted_num_correct = np.sum(weighted_correct_float)
    
        # Divide the number of weighted correct predictions by the sum of the
        # weights.
        accuracy = weighted_num_correct / sum_weights

        return accuracy, weighted_num_correct, sum_weights


    def test_model(self, generator, model):
        '''
        Input: 
            generator: an iterator instance that provides batches of inputs and targets
            model: a model instance 
        Output: 
            accuracy: float corresponding to the accuracy
        '''
        
        accuracy = 0.
        total_num_correct = 0
        total_num_pred = 0
        
        for batch in generator: 
            
            # Retrieve the inputs from the batch
            inputs = batch[0]
            
            # Retrieve the targets (actual labels) from the batch
            targets = batch[1]
            
            # Retrieve the example weight.
            example_weight = batch[2]

            # Make predictions using the inputs
            pred = model(inputs)
            
            # Calculate accuracy for the batch by comparing its predictions and targets
            batch_accuracy, batch_num_correct, batch_num_pred = self.compute_accuracy(pred, targets, example_weight)
            
            # Update the total number of correct predictions
            # by adding the number of correct predictions from this batch
            total_num_correct += batch_num_correct
            
            # Update the total number of predictions 
            # by adding the number of predictions made for the batch
            total_num_pred += batch_num_pred

        # Calculate accuracy over all examples
        accuracy = total_num_correct / total_num_pred

        return accuracy
