import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
# Reuters newspapers datasets, each belonging to one of 46 mutually exclusive topics
from keras.datasets import reuters

# how many unique words to use and validation set size
LEN_DICT = 12000
VAL_SET_SIZE = 1000

# one-hot encode reviews into a large vector where 1 indicates that the word is present
# take the LEN_DICT most frequent words and discard the rarer words
def vectorize_sequences(sequences, dimension=LEN_DICT):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# prints the original newswire
def print_newswire(newswire):
    word_index = reuters.get_word_index()
    word_lookup = dict([(value, key) for (key, value) in word_index.items()])
    print(' '.join([word_lookup.get(i - 3, '?') for i in newswire]))

# extract training and test data
# num_words=LEN_DICT extracts only the LEN_DICT most frequently used words; the rest is discarded
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=LEN_DICT)

# prepare input data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# prepare target data
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

x_val = x_train[:VAL_SET_SIZE]
y_val = one_hot_train_labels[:VAL_SET_SIZE]
partial_x_train = x_train[VAL_SET_SIZE:]
partial_y_train = one_hot_train_labels[VAL_SET_SIZE:]

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(LEN_DICT,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

results = model.evaluate(x_test, one_hot_test_labels)

# loss and accuracy of test data
print(results)

# training information
history_dict = history.history
acc_values = history.history['acc']
val_acc_values = history.history['val_acc']
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(acc_values) + 1)

# training and validation loss vs epoch
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# validation accuracy vs epoch
plt.subplot(1, 2, 2)
plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
