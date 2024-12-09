from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
from sklearn.metrics import confusion_matrix


#Models
def RandomForestModel(SpamData):
    
    SpamData['text'] = SpamData['text'].apply(lambda x: ' '.join(word.replace('\n', ' ') for word in x.split())) 
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  
    X = vectorizer.fit_transform(SpamData['text'])
   
    y = SpamData['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
  
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Step 7: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    precision = precision_score(y_test, y_pred)
    print(f'Precision: {precision:.2f}')
    
    # Calculate recall
    recall = recall_score(y_test, y_pred)
    print(f'Recall: {recall:.2f}')
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f'Confusion Matrix:\n{cm}')
    
    return model, y_pred, accuracy, precision, recall, cm


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
    
    precision = precision_score(y_test, y_pred)
    print(f"Precision: {precision:.2f}")
    
    # Calculate recall
    recall = recall_score(y_test, y_pred)
    print(f"Recall: {recall:.2f}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    return model, y_pred, accuracy, precision, recall, cm


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
     # Calculate recall
  recall = recall_score(y_test, predictions)
  print(f'Recall: {recall}')

    # Calculate confusion matrix
  cm = confusion_matrix(y_test, predictions)
  print(f'Confusion Matrix:\n{cm}')

  return model, predictions, test_accuracy, precision, recall, cm # Return the trained model







def MajorityVotingModel(SpamDataTest, rf_model, svm_model, nn_model):
    """
    This function makes predictions using three models (Random Forest, SVM, Neural Network) and combines their predictions 
    using majority voting to return the final prediction.
    """
    # Preprocess the test data (same as before)
    SpamDataTest['text'] = SpamDataTest['text'].apply(lambda x: ' '.join(word.replace('\n', ' ') for word in x.split()))
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    X_test = vectorizer.fit_transform(SpamDataTest['text'])
    
    # Step 2: Get predictions from all three models
    rf_pred = rf_model.predict(X_test)  # Random Forest prediction
    svm_pred = svm_model.predict(X_test)  # SVM prediction
    nn_pred = nn_model.predict(X_test)  # Neural Network prediction
    nn_pred = (nn_pred > 0.5).astype(int).flatten()
    
    
    # Step 3: Combine the predictions using majority voting
    # Stack the predictions vertically for easier voting
    predictions = np.column_stack((rf_pred, svm_pred, nn_pred))
    
    # Majority voting: Get the mode (most common value) for each instance
    final_pred = np.array([np.argmax(np.bincount(pred)) for pred in predictions])

    y_test = SpamDataTest["label"]
    
    accuracy = accuracy_score(y_test, final_pred)  # Accuracy calculation
    precision = precision_score(y_test, final_pred)  # Precision calculation
    recall = recall_score(y_test, final_pred)  # Recall calculation
    cm = confusion_matrix(y_test, final_pred)  # Confusion matrix
    
    # Print the results
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'Confusion Matrix:\n{cm}')
    
    return final_pred,accuracy,precision, recall, cm# Return the final predictions









#Other functions
def TestModel(SpamDataTest, model):
    SpamDataTest['text'] = SpamDataTest['text'].apply(lambda x: ' '.join(word.replace('\n', ' ') for word in x.split())) #Removing the \n and treating the words as 2 words
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # You can tune max_features
    X_test = vectorizer.fit_transform(SpamDataTest['text'])
    y_test= SpamDataTest["label"]
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    precision = precision_score(y_test, y_pred)
    print(f'Precision: {precision}')
    return y_pred #to save the predictions
    
    ##############
    
def SaveModel(model, name):
    model_pkl_file = name+".pkl"  
    with open(model_pkl_file, 'wb') as file:  
        pickle.dump(model, file)
                
    
def ReadModel(fileName):
    fileName=fileName+ ".pkl"
    with open(fileName, 'rb') as file:  
        model = pickle.load(file)
        return model
    
    
    ########### These currently for neural, dont really need to save vectorizer but in case

def SaveModel2(objects, filename): 
    # Save both the model and the vectorizer in the same file
    with open(filename + ".pkl", 'wb') as file:  
        pickle.dump(objects, file)
    print(f"{filename} saved successfully!")
    
def ReadModel2(fileName):
    with open(fileName+ ".pkl", 'rb') as file:
        saved_objects = pickle.load(file)
        model = saved_objects['model']
        vectorizer = saved_objects['vectorizer']
        return model,vectorizer