from sentence_transformers import SentenceTransformer
import torch
import nltk
from typing import Dict

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def split_into_sentences(text: str) -> list:
    """Split text into sentences using NLTK."""
    return nltk.sent_tokenize(text)

def calculate_semantic_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """Calculate cosine similarity between two texts using sentence embeddings."""
    emb1 = model.encode([text1], convert_to_tensor=True)
    emb2 = model.encode([text2], convert_to_tensor=True)
    return torch.nn.functional.cosine_similarity(emb1, emb2).item()

def evaluate_semantic(original_text: str, summary_text: str, model_name: str = "all-MiniLM-L6-v2") -> Dict[str, float]:
    """
    Evaluate semantic coverage of summary using sentence transformers.
    Returns both document-level and sentence-level similarity scores.
    """
    model = SentenceTransformer(model_name)
    
    # Document-level similarity
    doc_similarity = calculate_semantic_similarity(original_text, summary_text, model)
    
    # Sentence-level similarity
    orig_sentences = split_into_sentences(original_text)
    sum_sentences = split_into_sentences(summary_text)
    
    # Calculate max similarity for each summary sentence against original sentences
    sentence_similarities = []
    for sum_sent in sum_sentences:
        similarities = [calculate_semantic_similarity(orig_sent, sum_sent, model) 
                      for orig_sent in orig_sentences]
        sentence_similarities.append(max(similarities))
    
    # Average sentence-level similarity
    avg_sent_similarity = sum(sentence_similarities) / len(sentence_similarities) if sentence_similarities else 0
    
    return {
        "keyphrase_coverage_score": doc_similarity,  # Keep this key for compatibility
        "document_similarity": doc_similarity,
        "sentence_similarity": avg_sent_similarity
    } 