import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
# a set of 50,000 reviews from IMDB, each either a negative or positive review
# the data comes as a list of lists containing a number that corresponds to a
# specific word; the word-index dictionary is given by imdb.get_word_index()
from keras.datasets import imdb

# how many
LEN_DICT = 12000
VAL_SET_SIZE = 8000

# one-hot encode reviews into a large vector where 1 indicates that the word is present
# take the LEN_DICT most frequent words and discard the rarer words
def vectorize_sequences(sequences, dimension=LEN_DICT):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

word_index = imdb.get_word_index()
#word_lookup = dict([(value, key) for (key, value) in word_index.items()])

def print_review(review):
    word_index = imdb.get_word_index()
    word_lookup = dict([(value, key) for (key, value) in word_index.items()])
    print(' '.join([word_lookup.get(i - 3, '?') for i in review]))

# extract training and test data
# num_words=LEN_DICT extracts only the LEN_DICT most frequently used words; the rest is discarded
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=LEN_DICT)

# prepare input data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

print_review(train_data[0])

# convert labels to float32
y_train = train_labels.astype('float32')
y_test = test_labels.astype('float32')

# use the first VAL_SET_SIZE samples as the validation set
x_val = x_train[:VAL_SET_SIZE]
y_val = y_train[:VAL_SET_SIZE]
partial_x_train = x_train[VAL_SET_SIZE:]
partial_y_train = y_train[VAL_SET_SIZE:]

# build a neural network model
model = models.Sequential()
# two hidden layers each with 16 nodes and a 'relu' activation
# the first hidden layer takes in a vector of size 12000;
# this is a vector representing words that appear in the review
model.add(layers.Dense(16, activation='relu', input_shape=(LEN_DICT,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(
    # use root mean square to update neural network
    optimizer='rmsprop',
    # the loss function binary_crossentropy works very well for binary classification
    loss='binary_crossentropy',
    # keep track of the performance on the validation set throughout training
    metrics=['accuracy']
)

# train the model; use 6 epochs and a batch size of 512
# at the end of each epoch, validate on the validation set
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

# evaluate test set with model
results = model.evaluate(x_test, y_test)

# loss and accuracy of test data
print(results)

# model.fit returns an object containing information about the training
history_dict = history.history
acc_values = history.history['acc']
val_acc_values = history.history['val_acc']
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(acc_values) + 1)

# display a plot of training and validation loss vs epoch
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# display a plot of training and validation accuracy vs epoch
plt.subplot(1, 2, 2)
plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
