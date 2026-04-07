import pickle
import pandas as pd
import re
import nltk

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources (only first time)
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv("hotel_reviews.csv")   # change filename if needed

# Convert rating (1–5) → sentiment labels
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

# Apply preprocessing
X_clean = X.apply(clean_text)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Transform text
X_vec = vectorizer.transform(X_clean)

# Predict
y_pred = model.predict(X_vec)

# Debug check (optional but useful)
print("Actual labels:", y.unique())
print("Predicted labels:", set(y_pred))

# Evaluation Metrics
print("\n📊 MODEL EVALUATION RESULTS\n")

# Accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

# Classification Report
print("Classification Report:\n")
print(classification_report(y, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n")
print(confusion_matrix(y, y_pred))