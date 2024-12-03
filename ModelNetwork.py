import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def Neural_Model(SpamData):
     # Preprocess the text data
    SpamData['text'] = SpamData['text'].apply(lambda x: ' '.join(word.replace('\n', ' ') for word in x.split()))
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(SpamData['text'])
    y = SpamData['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = Sequential()
    model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))  # Input layer with ReLU activation
    model.add(Dropout(0.5))  # Dropout to prevent overfitting
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation

    # Step 6: Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Step 7: Train the model
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
    
    # Step 8: Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')

    # Step 9: Make predictions (optional)
    predictions = model.predict(X_test)
    predictions = (predictions > 0.5).astype(int)  # Convert probabilities to 0 or 1
    precision = precision_score(y_test, predictions)
    print(f'Precision: {precision}')

    print("Predictions:", predictions)
    
def test_model(model, data):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_new = vectorizer.transform(data['text']).toarray()
    
    predictions = model.predict(X_new)
    predictions = (predictions > 0.5).astype(int)  # Convert probabilities to 0 or 1
    
    df_new['predicted_label'] = predictions
    
    if 'label' in df_new.columns:
        y_true = df_new['label'].values
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions)
        
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        
    return df_new[['text', 'label', 'predicted_label']] if 'label' in df_new.columns else df_new[['text', 'predicted_label']]
    
def SaveModel(model, name):
    model_pkl_file = name+".pkl"  
    with open(model_pkl_file, 'wb') as file:  
        pickle.dump(model, file)    