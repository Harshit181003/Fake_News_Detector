import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = pickle.load(open('fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Preprocess input text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z ]", "", text)
    text = text.lower()
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.write("Paste a news article below and find out whether it's Real or Fake.")

user_input = st.text_area("Enter News Text Here", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        cleaned = clean_text(user_input)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]

        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.success(f"Prediction: **{label}**")
