# 🏨 Hotel Review Sentiment Analysis

This project performs multi-class sentiment classification (Negative, Neutral, Positive) on hotel reviews using Machine Learning and Natural Language Processing (NLP) techniques.

---

## 📌 Objective

To design, implement, evaluate, and deploy an end-to-end NLP system that classifies hotel reviews into sentiment categories.

---

## 🧠 Technologies Used

- Python
- Scikit-learn
- NLTK
- TF-IDF Vectorization
- Logistic Regression
- Support Vector Machine (SVM)
- Streamlit (for deployment)

---

## 📊 Features

- Text preprocessing (lowercasing, cleaning, stopword removal, lemmatization)
- TF-IDF feature extraction
- Multi-class classification (Negative, Neutral, Positive)
- Model comparison (Logistic Regression vs SVM)
- Confusion matrix visualization
- Error analysis of misclassified reviews
- Interactive web app using Streamlit

---

## 📁 Project Structure
Hotel_Review/
│
├── app.py
├── main.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md


---

## ⚙️ How to Run

### 1. Install dependencies
pip install -r requirements.txt


### 2. Train the model
python main.py


### 3. Run the Streamlit app
streamlit run app.py


---

## 🌐 Deployment

The model is deployed using Streamlit, providing an interactive interface where users can input hotel reviews and receive sentiment predictions.

---

## 📊 Results

- Logistic Regression achieved good performance on classification
- SVM provided comparable results
- Neutral class is relatively harder to classify due to ambiguity

---

## ⚠️ Limitations

- Difficulty handling negation (e.g., "not good")
- Mixed sentiment sentences may confuse the model
- Dataset imbalance may affect performance

---

## 🚀 Future Improvements

- Use deep learning models (LSTM, BERT)
- Improve handling of context and negation
- Use larger and balanced datasets

---

## 👩‍💻 Author

Aiswariya P Nair