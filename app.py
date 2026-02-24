import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Load a medium-sized English model (contains word vectors)
# Run 'python -m spacy download en_core_web_md' in terminal
nlp = spacy.load("en_core_web_md")

def calculate_accurate_score(resume_text, jd_text):
    """Uses NLP Word Vectors to find semantic similarity."""
    doc1 = nlp(resume_text)
    doc2 = nlp(jd_text)
    
    # Spacy's built-in similarity uses word vectors (Cosine Similarity)
    # This is much more accurate than TF-IDF for context
    score = doc1.similarity(doc2)
    return round(score * 100, 2)

def extract_keywords_nlp(jd_text):
    """Extracts only Nouns and Proper Nouns (Skills/Tech) from the JD."""
    doc = nlp(jd_text)
    keywords = set([token.text.lower() for token in doc 
                    if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop])
    return list(keywords)[:20]
