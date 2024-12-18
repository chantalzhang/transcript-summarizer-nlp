import json
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util

# Initialize KeyBERT and Sentence Transformer models
kw_model = KeyBERT()
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and fast embedding model

def load_preprocessed_json(file_path):
    """Load preprocessed sentences from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def compute_coverage_score(preprocessed_sentences, summary, ngram_range=(1, 2), percentage=10, similarity_threshold=0.7):
    """
    Compute the coverage score of a summary compared to preprocessed sentences using semantic similarity.

    Args:
        preprocessed_sentences (list): List of preprocessed sentences.
        summary (str): The summary to evaluate.
        ngram_range (tuple): Range of n-grams to extract.
        percentage (int): Percentage of sentences to determine 'top_n'.
        similarity_threshold (float): Minimum cosine similarity to consider a match.

    Returns:
        tuple: Coverage score, key phrases extracted, and matched phrases.
    """
    # Combine preprocessed sentences into a single source text
    source_text = " ".join(preprocessed_sentences)
    num_sentences = len(preprocessed_sentences)

    # Calculate top_n as a percentage of total sentences (minimum 5 phrases)
    top_n = max(5, (num_sentences * percentage) // 100)

    # Extract keyphrases using KeyBERT
    keyphrases = kw_model.extract_keywords(source_text, 
                                           keyphrase_ngram_range=ngram_range, 
                                           top_n=top_n)
    keywords = [kw[0] for kw in keyphrases]

    # Compute sentence embeddings
    summary_embedding = embedder.encode(summary, convert_to_tensor=True)
    keyword_embeddings = embedder.encode(keywords, convert_to_tensor=True)

    # Match key phrases based on cosine similarity
    matched_keywords = []
    for i, keyword in enumerate(keywords):
        similarity = util.cos_sim(summary_embedding, keyword_embeddings[i])
        if similarity.item() >= similarity_threshold:
            matched_keywords.append(keyword)

    coverage_score = len(matched_keywords) / len(keywords)
    return coverage_score, keywords, matched_keywords

def coverage_score_evaluation(preprocessed_file, summary, percentage=10, similarity_threshold=0.7):
    """
    Run coverage score evaluation on a summary using semantic similarity.

    Args:
        preprocessed_file (str): Path to the preprocessed sentences JSON file.
        summary (str): The summary to evaluate.
        percentage (int): Percentage of sentences to determine 'top_n'.
        similarity_threshold (float): Minimum cosine similarity to consider a match.

    Returns:
        dict: Coverage score, key phrases, and matched phrases.
    """
    preprocessed_sentences = load_preprocessed_json(preprocessed_file)
    coverage_score, keyphrases, matched_keywords = compute_coverage_score(
        preprocessed_sentences, summary, percentage=percentage, similarity_threshold=similarity_threshold
    )

    return {
        "Coverage Score": round(coverage_score, 2),
        "Keyphrases Extracted": keyphrases,
        "Matched Keyphrases": matched_keywords
    }
