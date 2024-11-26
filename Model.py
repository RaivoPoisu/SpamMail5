from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

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
