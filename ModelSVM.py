from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

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
    print(classification_report(y_test, y_pred))

    return model 