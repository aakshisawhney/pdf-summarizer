import streamlit as st
import fitz
from transformers import pipeline

st.title("⚡ Quick PDF Summarizer")

@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc[:1]:  # Only first page
            text += page.get_text()

    st.write("✅ Text extracted (first page only)")
    st.write(text[:300])

    if st.button("Summarize"):
        chunk = text[:1000]
        summary = summarizer(chunk, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
        st.subheader("📌 Summary")
        st.write(summary)



