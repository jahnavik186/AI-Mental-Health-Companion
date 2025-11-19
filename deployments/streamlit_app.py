import streamlit as st
from src.mental_health_bot import build_faiss_index, retrieve_similar, generate_response
import pandas as pd

# Load KB
kb = pd.read_csv('data/knowledge_base/articles.csv')
texts = kb['content'].tolist()
index, embeddings = build_faiss_index(texts)

st.title("🧠 AI Mental Health Companion")
user_name = st.text_input("Your Name")
if user_name:
    st.write(f"Hello {user_name}, I am here to listen and support you.")
    user_input = st.text_area("Share how you feel today")
    if st.button("Send"):
        retrieved_docs = retrieve_similar(user_input, texts, index, embeddings)
        context = "\n".join([doc for doc, _ in retrieved_docs])
        response = generate_response(user_input, context=context)
        st.text_area("AI Response", value=response, height=150)
