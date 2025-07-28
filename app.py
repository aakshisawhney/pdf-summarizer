import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF

# Load the summarization model (faster one)
@st.cache_resource
def load_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_model()

# Title
st.title("📄 PDF Summarizer App")
st.markdown("Upload a PDF and get a quick summary using ML!")

# File uploader
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Process PDF
if pdf_file is not None:
    with st.spinner("Extracting text from PDF..."):
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()

    # Display extracted text (optional)
    st.success("✅ Text Extracted")
    st.text_area("Extracted Text", full_text[:1000] + "..." if len(full_text) > 1000 else full_text, height=200)

    # Split text into chunks
    def split_text(text, max_chunk=1000):
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    chunks = split_text(full_text)
    st.info(f"🧠 Splitting into {len(chunks)} chunk(s)")

    # Summarize only first 2 chunks for speed
    if st.button("Summarize"):
        with st.spinner("Generating summary..."):
            summary = ""
            for chunk in chunks[:2]:  # Only summarize first 2 chunks
                s = summarizer(chunk, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
                summary += s + "\n\n"

            st.subheader("📌 Summary")
            st.write(summary)


