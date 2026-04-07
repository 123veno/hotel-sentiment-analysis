import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon') 

import gradio as gr
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer 
sia = SentimentIntensityAnalyzer()

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

def predict_sentiment(review):
    clean = clean_text(review)
    
    vec = vectorizer.transform([clean])
    ml_pred = model.predict(vec)[0]

    score = sia.polarity_scores(review)
    compound = score['compound']

    if compound >= 0.05:
        lex_pred = "Positive"
    elif compound <= -0.05:
        lex_pred = "Negative"
    else:
        lex_pred = "Neutral"

    # 🔥 FIXED LOGIC
    if ml_pred != lex_pred:
        final_pred = lex_pred
    else:
        final_pred = ml_pred

    emoji_map = {
        "Positive": "😊 Positive",
        "Negative": "😡 Negative",
        "Neutral": "😐 Neutral"
    }

    return emoji_map.get(final_pred, "😐 Neutral")

# Gradio UI
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="Enter your hotel review..."),
    outputs="text",
    title="🏨 Hotel Review Sentiment Analysis",
    description="Analyze hotel reviews using Machine Learning"
)

iface.launch()