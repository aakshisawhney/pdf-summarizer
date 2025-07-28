import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF

@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

st.title("📄 AI PDF Summarizer")
st.write("Upload a PDF file and get a summary using AI!")

pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

if pdf_file:
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    st.subheader("📑 Original Text (Preview)")
    st.text(full_text[:1000] + "..." if len(full_text) > 1000 else full_text)

    st.subheader("🧠 Summary")
    chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]
    final_summary = ""

    for chunk in chunks:
        summary = summarizer(chunk, max_length=60, min_length=20, do_sample=False)
        final_summary += summary[0]['summary_text'] + " "

    st.success(final_summary)
