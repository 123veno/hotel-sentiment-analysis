import pickle
import pandas as pd
import re
import nltk

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv("hotel_reviews.csv")

# Convert rating (1–5) → sentiment
def convert_rating(r):
    if r >= 4:
        return "Positive"
    elif r == 3:
        return "Neutral"
    else:
        return "Negative"

# Apply conversion
y = data['Rating'].apply(convert_rating)
X = data['Review']

# Preprocessing setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# -------------------------------
# ✅ Train-Test Split (IMPORTANT)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess ONLY test data
X_test_clean = X_test.apply(clean_text)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Transform test data
X_test_vec = vectorizer.transform(X_test_clean)

# Predict
y_pred = model.predict(X_test_vec)

# -------------------------------
# 📊 Evaluation Results
# -------------------------------
print("\n📊 MODEL EVALUATION RESULTS (Test Data)\n")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

# Classification Report
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
# -------------------------------
# 🔹 SVM Evaluation
# -------------------------------
svm_model = pickle.load(open("svm_model.pkl", "rb"))

svm_pred = svm_model.predict(X_test_vec)

print("\n📊 SVM MODEL RESULTS\n")

accuracy = accuracy_score(y_test, svm_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("Classification Report:\n")
print(classification_report(y_test, svm_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, svm_pred))