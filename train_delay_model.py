"""
Created on Monday Sep. 12-2022.

@author: Vasco Schiavo

How many times before heading to the station have we asked ourselves: but will my train be on time?
To answer this question, we developed a model using neural networks that consider the train station and
departure time of a train and predict its punctuality. This script contains the model to answer this question.
"""

# Packages and modules
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

def preprocessing_data(data, list_columns):
  """
  The method preprocesses the data before the model training.
  :param data: DataFrame
  :param list_columns: list of selected columns in data
  :return data: np.array
  """

  # Scaler for preprocessing
  scaler = MaxAbsScaler()

  # Preprocessing selected columns in the DataFrame
  for column in list_columns:
    data[column] = scaler.fit_transform(data[column].values.reshape(-1,1))


  return data

def creation_train_and_val_sets(data, split_size):
    """
    The method takes the raw data and proceeds to split it according to the parameter split_size into
    the train and validation set.
    :param data: np.array
    :param split_size: float between 0 and 1
    :return X_train, X_val, Y_train, Y_val: np.arrays
    """

    # Parameters for the split of the data set
    n_total = len(data)
    n_split = int(n_total * split_size)

    # Prints the size of the dataset and the split parameter
    print(f"The data set contains {n_total} training examples.")
    print(f"The rate decided for the split of the data set into training set and validation set is {split_size}.")

    # Randomizing the data
    np.random.shuffle(data)

    # Splits data into train and validation set
    X_train = np.array([vector[:-1] for vector in data[: n_split]])
    X_val = np.array([vector[:-1] for vector in data[n_split:]])

    Y_train = np.array([vector[-1] for vector in data[: n_split]])
    Y_val = np.array([vector[-1] for vector in data[n_split:]])

    # Prints the number of labeled examples of the training and validation set
    print("Therefore, there are")
    print(f"- {len(X_train)} labeled examples in the training set.")
    print(f"- {len(X_val)} labeled examples in the validation set. \n")

    return X_train, Y_train, X_val, Y_val

def model_delay_trains(X_train, Y_train, X_val, Y_val, epochs):    
    """
    The method defines the architecture of the model.
    :return history: list
    """

    # Defines a sequential NN
    model = tf.keras.Sequential([
        
        # Input layer
        tf.keras.layers.Dense(units=4, input_dim=4),

        # Hidden layers
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(units=10, activation = 'relu'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(units=10, activation = 'relu'),
        tf.keras.layers.Dropout(0.2),

        # Output layer
        tf.keras.layers.Dense(units=1, activation='linear')
    ])

    # Prints summary of the model
    model.summary()

    # Compiles the model
    model.compile(optimizer='adam', loss='mae')

    # Fits the model into the training set
    history = model.fit(X_train, Y_train, epochs=epochs, verbose=1, validation_data=(X_val, Y_val))

    return history

print("=================================================================")
print("                     TRAIN DELAY PREDICTION                      ")
print("=================================================================")

# Imports the data from a csv file 
data = pd.read_csv('/content/train_delay_data.csv', ',', index_col=0)

print("                      --- DATA INFORMATION ---                     ")

# Preprocessing the data
list_columns = ['Day of operation', 'Linie', 'Stop name', 'Departure time']
data = preprocessing_data(data, list_columns)

# Splitting the data into training and validation set
split_size = 0.8
X_train, Y_train, X_val, Y_val = creation_train_and_val_sets(data.to_numpy(), split_size)

print("                      --- MODEL TRAINING ---")

# Defines and trains the model
epochs = 100
history = model_delay_trains(X_train, Y_train, X_val, Y_val, epochs)

loss = history.history['loss']
val_loss = history.history['val_loss']

range_epochs = range(epochs)
start = 10

# Defines the graph of loss against epochs
plt.plot(range_epochs[start:], loss[start:], 'g', label='Loss')
plt.plot(range_epochs[start:], val_loss[start:], 'b', label='Val loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss/Val Loss')
plt.show()
