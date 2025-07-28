import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF

# Load the summarization model
@st.cache_resource
def load_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_model()

# Streamlit UI
st.title("📄 Full PDF Summarizer")
st.markdown("Upload a PDF and get a summary of the **entire document**!")

# Upload PDF
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Process PDF
if pdf_file is not None:
    with st.spinner("🔍 Extracting text..."):
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()

    st.success("✅ Text extracted from PDF")
    st.text_area("Extracted Text (preview)", full_text[:1000] + "..." if len(full_text) > 1000 else full_text, height=200)

    # Split text into chunks (each ~1000 characters)
    def split_text(text, max_chunk=1000):
        sentences = text.split('. ')
        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) <= max_chunk:
                current += sentence + ". "
            else:
                chunks.append(current.strip())
                current = sentence + ". "
        if current:
            chunks.append(current.strip())
        return chunks

    chunks = split_text(full_text)
    st.info(f"🧠 Document split into {len(chunks)} chunk(s) for summarization.")

    if st.button("Summarize Entire PDF"):
        with st.spinner("🧠 Summarizing... please wait (can take ~30-60 sec)"):
            full_summary = ""
            for i, chunk in enumerate(chunks):
                result = summarizer(chunk, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]
                full_summary += f"🔹 {result}\n\n"

            st.subheader("📌 Full PDF Summary")
            st.write(full_summary)




