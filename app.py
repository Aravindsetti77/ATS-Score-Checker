import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Backend Logic ---
def extract_text_from_pdf(file):
    """Extracts raw text from an uploaded PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def clean_text(text):
    """Basic NLP cleaning: lowercase, remove special chars/numbers."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

def calculate_ats_score(resume_text, jd_text):
    """Calculates similarity using TF-IDF and Cosine Similarity."""
    # Clean both texts
    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(jd_text)
    
    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_jd])
    
    # Compute Cosine Similarity (Result is between 0 and 1)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(similarity[0][0] * 100, 2)

def get_missing_keywords(resume_text, jd_text):
    """Simple set difference to find keywords in JD but not in Resume."""
    resume_words = set(clean_text(resume_text).split())
    jd_words = set(clean_text(jd_text).split())
    # You can expand this with a specific technical dictionary for better results
    missing = jd_words - resume_words
    # Filter for longer words (usually more meaningful skills)
    return [word for word in missing if len(word) > 3][:15]

# --- Frontend (Streamlit) ---
st.set_page_config(page_title="Pro ATS Checker", page_icon="📄", layout="centered")

st.title("🚀 Pro ATS Score Checker")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

with col2:
    job_description = st.text_area("Paste Job Description", height=200)

if st.button("Analyze Resume"):
    if uploaded_file and job_description:
        with st.spinner('Analyzing patterns...'):
            resume_text = extract_text_from_pdf(uploaded_file)
            
            if "Error" in resume_text:
                st.error(resume_text)
            else:
                score = calculate_ats_score(resume_text, job_description)
                missing = get_missing_keywords(resume_text, job_description)
                
                # Display Results
                st.subheader(f"ATS Match Score: {score}%")
                st.progress(score / 100)
                
                if score > 70:
                    st.success("Great match! Your resume is highly relevant to this role.")
                elif score > 40:
                    st.warning("Good start, but you're missing some key terms.")
                else:
                    st.error("Low match. Consider tailoring your resume more closely.")
                
                with st.expander("Keywords to consider adding:"):
                    st.write(", ".join(missing))
                    
    else:
        st.info("Please upload a PDF and paste the JD to begin.")

st.markdown("---")
st.caption("Built for Final Year Placement Prep | Built with Python & Scikit-Learn")