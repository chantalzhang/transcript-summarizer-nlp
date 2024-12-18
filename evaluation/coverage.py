from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import numpy as np

def extract_key_phrases(text, top_n_percent=1.0):
    """Extract key phrases from text using KeyBERT."""
    kw_model = KeyBERT()
    # Extract twice as many keywords as needed, then take top n%
    keywords = kw_model.extract_keywords(text, 
                                       keyphrase_ngram_range=(1, 3),
                                       stop_words='english',
                                       use_maxsum=True,
                                       nr_candidates=20)
    
    # Sort by score and take top n%
    keywords.sort(key=lambda x: x[1], reverse=True)
    num_to_keep = max(1, int(len(keywords) * top_n_percent))
    return [kw[0] for kw in keywords[:num_to_keep]]

def calculate_semantic_similarity(text1, text2, model_name='all-MiniLM-L6-v2'):
    """Calculate semantic similarity between two texts using sentence transformers."""
    model = SentenceTransformer(model_name)
    embedding1 = model.encode([text1])[0]
    embedding2 = model.encode([text2])[0]
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def evaluate_coverage(original_text, summary_text, similarity_threshold=0.5):
    """
    Evaluate how well the summary covers the key phrases from the original text.
    Returns a score between 0 and 1.
    """
    # Extract key phrases from original text
    key_phrases = extract_key_phrases(original_text)
    
    if not key_phrases:
        return {"keyphrase_coverage_score": 0.0}
    
    # Calculate coverage
    covered_phrases = 0
    for phrase in key_phrases:
        max_similarity = max(calculate_semantic_similarity(phrase, sent) 
                           for sent in summary_text.split('.'))
        if max_similarity >= similarity_threshold:
            covered_phrases += 1
    
    coverage_score = covered_phrases / len(key_phrases)
    return {"keyphrase_coverage_score": coverage_score} 