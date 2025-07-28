import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF

# Title
st.title("📄 AI PDF Summarizer")

# Load summarizer model
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Extract text
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()

    st.write("✅ Text Extracted")
    st.write(text[:500] + "..." if len(text) > 500 else text)

    if st.button("Generate Summary"):
        # Split into chunks
        def split_text(text, max_chunk=1000):
            sentences = text.split('. ')
            chunks = []
            chunk = ""
            for sentence in sentences:
                if len(chunk) + len(sentence) <= max_chunk:
                    chunk += sentence + ". "
                else:
                    chunks.append(chunk.strip())
                    chunk = sentence + ". "
            chunks.append(chunk.strip())
            return chunks

        chunks = split_text(text)
        st.write(f"🧠 Splitting into {len(chunks)} chunk(s)")

        # Generate summary
        final_summary = ""
        for chunk in chunks:
            summary = summarizer(chunk, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
            final_summary += summary + " "

        # Display
        st.subheader("📌 Summary")
        st.write(final_summary.strip())

