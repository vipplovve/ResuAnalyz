import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing, metrics
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

def ReshapeForLSTM(features):
    return features.reshape(features.shape[0], 1, features.shape[1])

def LossPlot(History):
    Figure, Axis = plt.subplots(1, 1)
    Axis.plot(History.history['loss'], label='Training Loss')
    Axis.plot(History.history['val_loss'], label='Validation Loss')
    Axis.set_xlabel('Epoch')
    Axis.set_ylabel('Loss')
    Axis.grid(True)
    plt.legend()
    plt.show()

FileName = input("Enter the name of the DataSet: ")
Categorials = input("Enter the name of All Categorical Variables & The Target Variable: ").split()
Dataset = pd.read_csv(FileName)
Dataset = pd.get_dummies(Dataset, columns=Categorials)

Train, Valid, Test = np.split(Dataset.sample(frac=1), [int(.6*len(Dataset)), int(.8*len(Dataset))])

TrainF, TrainO = Partition(Train)
ValidF, ValidO = Partition(Valid)
TestF, TestO = Partition(Test)

TrainF = ReshapeForLSTM(TrainF)
ValidF = ReshapeForLSTM(ValidF)
TestF = ReshapeForLSTM(TestF)

neurons = int(input("Enter the number of neurons for LSTM layers: "))
dropout_probability = float(input("Enter the dropout probability (between 0.2 and 0.5): "))
epochs = int(input("Enter the number of epochs for training: "))

inputs = tf.keras.Input(shape=(TrainF.shape[1], TrainF.shape[2]))
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(neurons, return_sequences=True))(inputs)
x = tf.keras.layers.Dropout(dropout_probability)(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(neurons))(x)
x = tf.keras.layers.Dropout(dropout_probability)(x)
x = tf.keras.layers.Dense(neurons, activation='relu')(x)
x = tf.keras.layers.Dropout(dropout_probability)(x)
outputs = tf.keras.layers.Dense(8, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

for epoch in range(epochs):
    print(f"\nTraining Epoch {epoch + 1}/{epochs}")
    history = model.fit(TrainF, TrainO, validation_data=(ValidF, ValidO), epochs=1, verbose=0)

    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    
    print(f"Training Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

    val_predictions = model.predict(ValidF)
    val_pred_labels = np.argmax(val_predictions, axis=1)
    val_true_labels = np.argmax(ValidO, axis=1)

    cm = metrics.confusion_matrix(val_true_labels, val_pred_labels, labels=[0, 1, 2])

Predictions = model.predict(TestF)
Predictions = np.argmax(Predictions, axis=1)
TestO_labels = np.argmax(TestO, axis=1)

Accuracy = metrics.accuracy_score(TestO_labels, Predictions)
print("\nAccuracy Score For This BiLSTM Model is: ", Accuracy * 100, "%")

save_model = input("\nDo You Wish To Save This Model? Enter 'Yes' or 'No': ")
if save_model.lower() == 'yes':
    model.save('BiLSTM.keras')
    print("\nModel Saved Successfully!")
else:
    print("\nModel Not Saved!")