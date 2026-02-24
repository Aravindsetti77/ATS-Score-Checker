import streamlit as st
import pdfplumber
import spacy

# 1. Load the model safely
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        # Fallback if the download failed
        import os
        os.system("python -m spacy download en_core_web_md")
        return spacy.load("en_core_web_md")

nlp = load_nlp()

def get_pro_text(file):
    """Uses pdfplumber for better layout handling."""
    with pdfplumber.open(file) as pdf:
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def calculate_match(resume, jd):
    """Semantic similarity: understands 'Python' is related to 'Coding'."""
    doc_resume = nlp(resume)
    doc_jd = nlp(jd)
    # This uses word vectors, not just word counts
    score = doc_resume.similarity(doc_jd)
    return round(score * 100, 2)

# --- UI Logic ---
st.title("🎯 Precise ATS Analyzer")

uploaded = st.file_uploader("Upload PDF", type="pdf")
jd_text = st.text_area("Paste Job Description")

if st.button("Analyze") and uploaded and jd_text:
    resume_text = get_pro_text(uploaded)
    score = calculate_match(resume_text, jd_text)
    
    st.metric("Match Accuracy", f"{score}%")
    
    if score < 50:
        st.error("Too generic. Add specific tech stack keywords.")
    else:
        st.success("Solid match!")
