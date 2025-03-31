import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load model
model = joblib.load('fake_news_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Text cleaning
def clean_text(text):
    ps = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Streamlit UI
st.title("🔍 Fake News Detector")
user_input = st.text_area("Enter news headline or text:")
if st.button("Predict"):
    cleaned_text = clean_text(user_input)
    vector = tfidf.transform([cleaned_text])
    prediction = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    if prediction == 0:
        st.error(f"⚠️ FAKE NEWS (Confidence: {proba[0]*100:.1f}%)")
    else:
        st.success(f"✅ REAL NEWS (Confidence: {proba[1]*100:.1f}%)")
