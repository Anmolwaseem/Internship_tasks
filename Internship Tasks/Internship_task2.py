import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import string

# -----------------
# Step 1: Load Data
# -----------------
train_data = pd.read_csv(r"D:\Internship Tasks\train.csv")
test_data = pd.read_csv(r"D:\Internship Tasks\test.csv")

print("Train dataset shape:", train_data.shape)
print("Test dataset shape:", test_data.shape)

print("\nTrain dataset sample:\n", train_data.head())
print("\nTest dataset sample:\n", test_data.head())

# -----------------
# Step 2: Preprocess Text
# -----------------
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return text

train_data['text'] = train_data['text'].astype(str).apply(preprocess_text)
test_data['text'] = test_data['text'].astype(str).apply(preprocess_text)

# -----------------
# Step 3: Vectorization (CountVectorizer) 
# -----------------
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(train_data['text'])
y_train = train_data['sentiment']

X_test = vectorizer.transform(test_data['text'])
y_test = test_data['sentiment']

# -----------------
# Step 4: Train Model
# -----------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# -----------------
# Step 5: Evaluate Model
# -----------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy on Test Data:", accuracy)
print("\nSample Predictions:")
print(list(zip(test_data['text'][:10], y_pred[:10])))
