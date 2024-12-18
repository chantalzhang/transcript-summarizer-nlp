def evaluate_compression(original_text, summary_text):
    """
    Calculate compression ratio between original text and summary.
    Returns a score representing how concise the summary is (lower is more concise).
    """
    original_words = len(original_text.split())
    summary_words = len(summary_text.split())
    
    if original_words == 0:
        return {"compression_score": 0.0}
    
    compression_ratio = summary_words / original_words
    return {"compression_score": compression_ratio} 