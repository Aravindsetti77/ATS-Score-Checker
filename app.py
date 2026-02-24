import streamlit as st
import pdfplumber
import re
from collections import Counter
import math

# --- Precision Logic (No heavy libraries) ---
def get_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def clean_and_tokenize(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return [word for word in text.split() if len(word) > 2]

def calculate_cosine_similarity(text1, text2):
    """Accurate manual cosine similarity for stability."""
    vec1 = Counter(clean_and_tokenize(text1))
    vec2 = Counter(clean_and_tokenize(text2))
    
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    return round((numerator / denominator) * 100, 2) if denominator else 0.0

# --- Streamlit UI ---
st.set_page_config(page_title="Ultra-Stable ATS", layout="wide")
st.title("🛡️ Ultra-Stable ATS Checker")

# Use columns to prevent the "wall of text" look
col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
with col2:
    jd_input = st.text_area("Paste Job Description", height=200)

if st.button("Run Accuracy Analysis"):
    if resume_file and jd_input:
        try:
            resume_content = get_text_from_pdf(resume_file)
            score = calculate_cosine_similarity(resume_content, jd_input)
            
            st.divider()
            st.header(f"Match Score: {score}%")
            
            if score > 60:
                st.balloons()
                st.success("Strong match! Your resume aligns well with the keywords.")
            else:
                st.warning("Needs improvement. Try adding more specific industry terms.")
                
        except Exception as e:
            st.error(f"Something went wrong: {e}")
    else:
        st.info("Please upload a PDF and paste a JD first.")
