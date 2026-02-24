import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- Pro Functions ---
def extract_text_pro(file):
    """Uses pdfplumber to handle complex resume layouts accurately."""
    try:
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
            return text
    except Exception as e:
        return f"Error: {e}"

def clean_text_pro(text):
    text = text.lower()
    text = re.sub(r'http\S+\s*', ' ', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', ' ', text)     # Remove punctuation
    return text

def calculate_pro_score(resume, jd):
    # Using ngram_range=(1, 2) allows the AI to recognize 2-word skills like "Machine Learning"
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([clean_text_pro(resume), clean_text_pro(jd)])
    similarity = cosine_similarity(matrix[0:1], matrix[1:2])
    return round(float(similarity[0][0]) * 100, 2)

# --- Streamlit UI ---
st.set_page_config(page_title="Elite ATS Checker", page_icon="🎯")
st.title("🎯 Pro ATS Matcher")
st.write("This version uses N-gram phrase matching for higher accuracy.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
jd_text = st.text_area("Paste Job Description", height=250)

if st.button("Analyze Match"):
    if uploaded_file and jd_text:
        with st.spinner("Processing documents..."):
            resume_text = extract_text_pro(uploaded_file)
            
            if "Error" in resume_text:
                st.error(resume_text)
            else:
                score = calculate_pro_score(resume_text, jd_text)
                
                st.metric(label="Match Rate", value=f"{score}%")
                st.progress(score / 100)
                
                if score > 75:
                    st.success("High Match! Your resume looks great for this role.")
                elif score > 50:
                    st.warning("Average Match. Try adding more specific keywords from the JD.")
                else:
                    st.error("Low Match. You need to tailor your resume significantly.")
    else:
        st.info("Please provide both a resume and a job description.")
