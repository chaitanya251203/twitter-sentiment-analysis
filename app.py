import streamlit as st
import joblib
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np

# Load saved model and vectorizer
model = joblib.load("trained_model.sav")
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

nltk.download('stopwords')
port_stem = PorterStemmer()

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [port_stem.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet below to analyze its sentiment:")

user_input = st.text_area("Tweet Text", "Type your tweet here...")

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a tweet before analyzing.")
    else:
        processed_text = preprocess_text(user_input)
        transformed_text = vectorizer.transform([processed_text])
        prediction = model.predict(transformed_text)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.write(f"Sentiment: **{sentiment}**")
