from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

#Models
def RandomForestModel(SpamData):
    
    SpamData['text'] = SpamData['text'].apply(lambda x: ' '.join(word.replace('\n', ' ') for word in x.split())) #Removing the \n and treating the words as 2 words
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # You can tune max_features
    X = vectorizer.fit_transform(SpamData['text'])
   
# Step 3: Define target variable
    y = SpamData['label']
    
    # Step 4: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Step 5: Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Step 6: Make predictions
    y_pred = model.predict(X_test)
    
    # Step 7: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    return model


def SVM_Model(SpamData):
    # Preprocess the text data
    SpamData['text'] = SpamData['text'].apply(lambda x: ' '.join(word.replace('\n', ' ') for word in x.split()))
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(SpamData['text'])
    y = SpamData['label']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train the SVM model
    model = SVC(kernel='linear')
  # Linear kernel is often suitable for text classification
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy:   {accuracy * 100:.2f}%")

    return model


def Neural_Model(SpamData):
    # Preprocess the text data
  SpamData['text'] = SpamData['text'].apply(lambda x: ' '.join(word.replace('\n', ' ') for word in x.split()))
  vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
  X = vectorizer.fit_transform(SpamData['text'])
  y = SpamData['label']

  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  # Define the neural network model
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(512, input_shape=(X_train.shape[1],), activation='relu'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  # Compile the model
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

  # Train the model
  history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

  # Evaluate the model on test data
  test_loss, test_accuracy = model.evaluate(X_test, y_test)
  print(f'Test Loss: {test_loss}')
  print(f'Test Accuracy: {test_accuracy}')

  # (Optional) Make predictions
  predictions = model.predict(X_test)
  predictions = (predictions > 0.5).astype(int)
  precision = precision_score(y_test, predictions)
  print(f'Precision: {precision}')
  print("Predictions:", predictions)

  return model  # Return the trained model







#Other functions
def TestModel(SpamDataTest, model):
    SpamDataTest['text'] = SpamDataTest['text'].apply(lambda x: ' '.join(word.replace('\n', ' ') for word in x.split())) #Removing the \n and treating the words as 2 words
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # You can tune max_features
    X_test = vectorizer.fit_transform(SpamDataTest['text'])
    y_test= SpamDataTest["label"]
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
def SaveModel(model, name):
    model_pkl_file = name+".pkl"  
    with open(model_pkl_file, 'wb') as file:  
        pickle.dump(model, file)

    
def ReadModel(fileName):
    fileName=fileName+ ".pkl"
    with open(fileName, 'rb') as file:  
        model = pickle.load(file)
        return model
