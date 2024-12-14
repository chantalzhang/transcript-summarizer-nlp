import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def calculate_precision_recall_f1(extracted, reference):
    extracted_set = set(extracted)
    reference_set = set(reference)
    true_positives = len(extracted_set & reference_set)
    false_positives = len(extracted_set - reference_set)
    false_negatives = len(reference_set - extracted_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score

def calculate_cosine_similarity(sentences):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    similarity_matrix = cosine_similarity(embeddings)
    avg_similarity = np.mean(similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)])
    return avg_similarity

def extract_keywords_tfidf(text, num_keywords=10):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=num_keywords)
    tfidf_matrix = vectorizer.fit_transform([text])
    return list(vectorizer.get_feature_names_out())
