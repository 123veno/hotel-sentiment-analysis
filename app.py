import nltk
nltk.download('stopwords')
nltk.download('wordnet')

import streamlit as st
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# UI
st.title("🏨 Hotel Review Sentiment Analysis")
st.write("Analyze hotel reviews using Machine Learning")

review = st.text_area("✍️ Enter your review:", height=150)

model_choice = st.selectbox("Choose Model", ["Logistic Regression", "SVM"])

if st.button("Predict"):
    clean = clean_text(review)
    vec = vectorizer.transform([clean])
    
    prediction = model.predict(vec)[0]  # you can add SVM later

    if prediction == "Positive":
        st.success(f"😊 Sentiment: {prediction}")
    elif prediction == "Negative":
        st.error(f"😡 Sentiment: {prediction}")
    else:
        st.warning(f"😐 Sentiment: {prediction}")