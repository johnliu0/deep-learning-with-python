import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
# boston housing prices, in thousands
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# normalize all features such that the mean is 0 and the standard deviation is one
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std
# note that the test data is normalized using mean and std computed from training data
test_data -= mean
test_data /= std

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    # measure mean absolute error; the absolute difference between predictions and targets
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# use k-fold validation to evaluate network
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_mae_histories = []
for i in range(k):
    print(f'Processing fold: {i}')
    val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]],
        axis=0
    )

    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
        train_targets[(i + 1) * num_val_samples:]],
        axis=0
    )

    model = build_model()
    history = model.fit(
        partial_train_data, partial_train_targets, epochs=num_epochs,
        batch_size=1,
        validation_data=(val_data, val_targets)
    )

    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

print(all_mae_histories)
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
print(f'Mean MAE: {average_mae_history}')

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
