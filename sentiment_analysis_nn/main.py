'''
How to run:
    In the terminal, go to this folder (sentiment_analysis_nn), make sure your docker desktop is running and:
        docker build --tag custom_dnn .
        docker run custom_dnn

'''

from dnn import DNN
import random as rnd

dnn_obj = DNN()



##################################################################################################
# Creation of a new model
##################################################################################################

# Load positive and negative tweets to be used in the training and evaluation of the model
all_positive_tweets, all_negative_tweets = dnn_obj.load_tweets_from_nltk()

train_pos  = all_positive_tweets[:4000] # generating training set for positive tweets
val_pos   = all_positive_tweets[4000:] # generating validation set for positive tweets

train_neg  = all_negative_tweets[:4000] # generating training set for negative tweets
val_neg   = all_negative_tweets[4000:] # generating validation set for negative tweets

# Combine training data into one set
train_dataset = train_pos + train_neg
val_dataset  = val_pos + val_neg

# Build the vocabulary 
vocab = dnn_obj.build_vocabulary(dataset=train_dataset)
# Creation of the model
model = dnn_obj.classifier(vocab_size=len(vocab))


##################################################################################################
# Training the model
##################################################################################################

# Set the random number generator for the shuffle procedure
rnd.seed(30) 

# Uncomment to see the output of train generator
#dnn_obj.see_output_of_train_generator()

# Training loop contains the trained model (using cross validation through n_steps)
training_loop = dnn_obj.train_model(train_pos=train_pos, train_neg=train_neg, val_pos=val_pos, val_neg=val_neg, vocab=vocab, classifier=model, n_steps=100)


# Print expected outputs of some phrases vs sentiment given by trained model
#dnn_obj.validate_predictions_trained_model(training_loop=training_loop)


##################################################################################################
# Test predictions of the trained model
##################################################################################################
tmp_val_generator = dnn_obj.val_generator(val_pos=val_pos, val_neg=val_neg, vocab=vocab, batch_size=64)

# get one batch
tmp_batch = next(tmp_val_generator)

# Position 0 has the model inputs (tweets as tensors)
# position 1 has the targets (the actual labels)
tmp_inputs, tmp_targets, tmp_example_weights = tmp_batch

# feed the tweet tensors into the model to get a prediction
tmp_pred = training_loop.eval_model(tmp_inputs)

tmp_acc, tmp_num_correct, tmp_num_predictions = dnn_obj.compute_accuracy(tmp_pred, tmp_targets, tmp_example_weights)

print(f"Model's prediction accuracy on a single training batch is: {100 * tmp_acc}%")
print(f"Weighted number of correct predictions {tmp_num_correct}; weighted number of total observations predicted {tmp_num_predictions}")

model = training_loop.eval_model
accuracy = dnn_obj.test_model(dnn_obj.test_generator(batch_size=16, val_pos=val_pos, val_neg=val_neg, vocab=vocab), model)

print(f'The accuracy of your model on the validation set is {accuracy:.4f}', )


##################################################################################################
# Using the model for Sentiment Analysis
##################################################################################################~


#If we aren't training the model in real time, let's load the trained model from a file
#weights, state = model.init_from_file(PATH)

#Missing init, este init pode ser um modelo já guardado, ou o modelo que treinamos nas linhas anteriores 


sentence = "HEY, iphone 13 is great :)"
preds, sentiment = dnn_obj.predict(sentence=sentence, vocab=vocab, model=model)

print(f"Phrase: {sentence}\n Sentiment: {sentiment}")