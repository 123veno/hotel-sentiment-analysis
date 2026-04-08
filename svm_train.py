import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv("hotel_reviews.csv")

# Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

data["clean_review"] = data["Review"].apply(clean_text)

# Convert ratings
def convert_rating(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"

data["sentiment"] = data["Rating"].apply(convert_rating)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data["clean_review"], data["sentiment"],
    test_size=0.2, random_state=42
)

# Load existing vectorizer (IMPORTANT)
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Transform
X_train_vec = vectorizer.transform(X_train)

# Train SVM
svm_model = SVC()
svm_model.fit(X_train_vec, y_train)

# Save model
pickle.dump(svm_model, open("svm_model.pkl", "wb"))

print("✅ SVM model trained and saved successfully!")