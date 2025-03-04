import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plotter
import os

from sklearn import preprocessing
from sklearn import utils
from sklearn import metrics

import tensorflow as tf

os.system('cls')

def Partition(dataframe, Oversample=False):
    
    Features = dataframe.drop(columns=[col for col in dataframe.columns if col.startswith('Positions_')]).values;

    Output = dataframe[[col for col in dataframe.columns if col.startswith('Positions_')]].values;

    Scaler = sklearn.preprocessing.StandardScaler();

    Features = Scaler.fit_transform(Features);

    if Oversample:

        Oversampler = sklearn.utils.resample;

        Features, Output = Oversampler(Features, Output);

    return Features, Output;

def LossPlot(History):
    
    Figure, Axis = plotter.subplots(1, 1)
    Axis.plot(History.history['loss'], label='Training Loss')
    Axis.plot(History.history['val_loss'], label='Validation Loss')
    Axis.set_xlabel('Epoch')
    Axis.set_ylabel('Binary Cross Entropy Loss')
    Axis.grid(True)
    plotter.legend()
    plotter.show()

def AccuracyPlot(History):
    
    Figure, Axis = plotter.subplots(1, 1)
    Axis.plot(History.history['accuracy'], label='Training Accuracy')
    Axis.plot(History.history['val_accuracy'], label='Validation Accuracy')
    Axis.set_xlabel('Epoch')
    Axis.set_ylabel('Accuracy')
    Axis.grid(True)
    plotter.legend()
    plotter.show()

FileName = input("Enter the name of the DataSet: ")
print()
Categorials = input("Enter the name of All Categorial Variables & The Target Variable: ").split()
print()

Dataset = pd.read_csv(FileName)
Dataset = pd.get_dummies(Dataset, columns=Categorials)

Train, Valid, Test = np.split(Dataset.sample(frac=1), [int(.6*len(Dataset)), int(.8*len(Dataset))])

os.system('cls')

print("The Dataset: -\n")
print(Dataset.head(), '\n')

TrainF, TrainO = Partition(Train)
ValidF, ValidO = Partition(Valid)
TestF, TestO = Partition(Test)

print("Creating a Dense NN Model: -\n")
neurons = int(input("Enter No. of Nodes for Each Layer: "))
print()
dropout_probability = float(input("Enter the Dropout Probability: "))
print()
Epochs = int(input("Enter the Number of Epochs: "))
print()

NNModel = tf.keras.models.Sequential([
    tf.keras.layers.Dense(neurons, activation='relu'),
    tf.keras.layers.Dropout(dropout_probability),
    tf.keras.layers.Dense(neurons, activation='relu'),
    tf.keras.layers.Dropout(dropout_probability),
    tf.keras.layers.Dense(neurons, activation='relu'),
    tf.keras.layers.Dropout(dropout_probability),
    tf.keras.layers.Dense(len([col for col in Dataset.columns if col.startswith('Positions_')]), activation='softmax')
])

NNModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']);

History = NNModel.fit(TrainF, TrainO, epochs=Epochs, validation_data=(ValidF, ValidO))

LossPlot(History)
AccuracyPlot(History)

Predictions = NNModel.predict(TestF)

print(Predictions);

Predictions = np.round(Predictions)

print("\nPredictions: -\n")
Report = sklearn.metrics.classification_report(TestO, Predictions)
print("\nClassification Report using a Dense Model: -\n")
print(Report)

Accuracy = sklearn.metrics.accuracy_score(TestO, Predictions)

print("Specifications For The Network: \n")
print(NNModel.summary(), '\n')
print("\nAccuracy Score For This Dense Model is: ", Accuracy * 100, "%", '\n')

print("Do You Wish To Save This Model?")
input = input("Enter 'Yes' or 'No': ")

if input == 'Yes':
    NNModel.save('NNDense.keras')
    print("\nModel Saved Successfully!")
else:
    print("\nModel Not Saved!")