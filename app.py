import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import torch

# --- Settings ---
st.set_page_config(page_title="Pro ATS Analyzer", page_icon="🤖")

# Cache the model so it only loads ONCE (saves memory and time)
@st.cache_resource
def load_bert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_bert_model()

def extract_text(file):
    with pdfplumber.open(file) as pdf:
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# --- UI ---
st.title("🚀 Pro ATS Similarity Engine")
st.info("Using BERT Neural Networks for high-accuracy semantic matching.")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
with col2:
    jd_text = st.text_area("Paste Job Description", height=200)

if st.button("Analyze Real Match"):
    if uploaded_file and jd_text:
        with st.spinner("AI is thinking..."):
            resume_text = extract_text(uploaded_file)
            
            # Convert text to "embeddings" (mathematical meaning)
            resume_vec = model.encode(resume_text, convert_to_tensor=True)
            jd_vec = model.encode(jd_text, convert_to_tensor=True)
            
            # Calculate cosine similarity
            score = util.cos_sim(resume_vec, jd_vec)
            final_score = round(float(score[0][0]) * 100, 2)
            
            st.divider()
            st.subheader(f"Semantic Match Score: {final_score}%")
            st.progress(final_score / 100)
            
            if final_score > 70:
                st.success("High Relevance! Your experience matches the core intent of this JD.")
            elif final_score > 40:
                st.warning("Moderate Match. Consider aligning your phrasing more closely.")
            else:
                st.error("Low Match. This resume might not pass the initial AI screening.")
    else:
        st.warning("Please upload a file and paste the job description.")
