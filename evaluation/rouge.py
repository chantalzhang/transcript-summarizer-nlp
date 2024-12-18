from collections import Counter
import nltk
from typing import Dict, List, Tuple

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def get_ngrams(text: str, n: int) -> List[Tuple[str, ...]]:
    """Convert text into n-grams."""
    tokens = nltk.word_tokenize(text.lower())
    return list(nltk.ngrams(tokens, n))

def calculate_rouge_n(reference_text: str, summary_text: str, n: int = 1) -> Dict[str, float]:
    """
    Calculate ROUGE-N score between reference and summary texts.
    Args:
        reference_text: The original text
        summary_text: The summary to evaluate
        n: The n-gram size (1 for unigrams, 2 for bigrams)
    Returns:
        Dictionary containing ROUGE precision, recall, and F1 scores
    """
    # Get n-grams for both texts
    ref_ngrams = Counter(get_ngrams(reference_text, n))
    sum_ngrams = Counter(get_ngrams(summary_text, n))
    
    # Find overlapping n-grams
    overlap_ngrams = ref_ngrams & sum_ngrams
    
    # Calculate counts
    overlap_count = sum(overlap_ngrams.values())
    ref_count = sum(ref_ngrams.values())
    sum_count = sum(sum_ngrams.values())
    
    # Calculate precision and recall
    precision = overlap_count / sum_count if sum_count > 0 else 0
    recall = overlap_count / ref_count if ref_count > 0 else 0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        f"rouge_{n}_precision": precision,
        f"rouge_{n}_recall": recall,
        f"rouge_{n}_f1": f1
    }

def evaluate_rouge(original_text: str, summary_text: str) -> Dict[str, float]:
    """
    Evaluate using both ROUGE-1 and ROUGE-2 scores.
    Returns combined scores dictionary.
    """
    # Calculate both ROUGE-1 and ROUGE-2 scores
    rouge1_scores = calculate_rouge_n(original_text, summary_text, n=1)
    rouge2_scores = calculate_rouge_n(original_text, summary_text, n=2)
    
    # Combine scores
    rouge_scores = {
        "keyphrase_coverage_score": rouge1_scores["rouge_1_f1"],  # Keep this key for compatibility
        **rouge1_scores,
        **rouge2_scores
    }
    
    return rouge_scores 