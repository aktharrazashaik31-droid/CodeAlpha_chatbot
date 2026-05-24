import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🤖 FAQ Chatbot")

faq_questions = [
    "What is AI?",
    "What is Python?",
    "What is machine learning?",
    "What is Streamlit?",
    "Who developed Python?"
]

faq_answers = [
    "AI stands for Artificial Intelligence.",
    "Python is a popular programming language.",
    "Machine learning is a branch of AI.",
    "Streamlit is a Python library for web apps.",
    "Python was developed by Guido van Rossum."
]

user_question = st.text_input("Ask a question")

if st.button("Get Answer"):

    questions = faq_questions + [user_question]

    vectorizer = TfidfVectorizer()

    tfidf = vectorizer.fit_transform(questions)

    similarity = cosine_similarity(tfidf[-1], tfidf[:-1])

    index = similarity.argmax()

    st.success(faq_answers[index])