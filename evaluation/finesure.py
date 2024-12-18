import json
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load entailment model for Faithfulness
entailment_model = pipeline("text-classification", model="facebook/bart-large-mnli")

def load_preprocessed_json(file_path):
    """Load preprocessed sentences from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def compute_faithfulness(preprocessed_sentences, summary):
    """
    Compute faithfulness of a summary compared to the preprocessed sentences.
    Faithfulness is measured using a text entailment model.
    """
    entailment_scores = []
    for sentence in preprocessed_sentences:
        result = entailment_model(sentence, candidate_labels=[summary])
        entailment_scores.append(result[0]['score'])
    return sum(entailment_scores) / len(entailment_scores)

def compute_completeness(preprocessed_sentences, summary):
    """
    Compute completeness of a summary compared to preprocessed sentences.
    Completeness is measured using TF-IDF cosine similarity.
    """
    vectorizer = TfidfVectorizer()
    source_text = " ".join(preprocessed_sentences)
    tfidf_matrix = vectorizer.fit_transform([source_text, summary])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

def compute_conciseness(preprocessed_sentences, summary):
    """
    Compute conciseness of a summary compared to preprocessed sentences.
    Conciseness is the compression ratio (input words / summary words).
    """
    source_word_count = sum(len(sentence.split()) for sentence in preprocessed_sentences)
    summary_word_count = len(summary.split())
    return source_word_count / summary_word_count

def finesure_evaluation(preprocessed_file, summary):
    """
    Run FineSurE evaluation (Faithfulness, Completeness, Conciseness) on a summary.
    """
    preprocessed_sentences = load_preprocessed_json(preprocessed_file)
    faithfulness = compute_faithfulness(preprocessed_sentences, summary)
    completeness = compute_completeness(preprocessed_sentences, summary)
    conciseness = compute_conciseness(preprocessed_sentences, summary)

    return {
        "Faithfulness": round(faithfulness, 2),
        "Completeness": round(completeness, 2),
        "Conciseness (Compression Ratio)": round(conciseness, 2)
    }
