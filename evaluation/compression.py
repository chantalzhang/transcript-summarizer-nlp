import json

def load_preprocessed_json(file_path):
    """Load preprocessed sentences from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def compute_compression_ratio(preprocessed_sentences, summary):
    """
    Compute the compression ratio of a summary compared to preprocessed sentences.
    Compression Ratio = Total Words in Source / Total Words in Summary
    """
    source_word_count = sum(len(sentence.split()) for sentence in preprocessed_sentences)
    summary_word_count = len(summary.split())
    return source_word_count / summary_word_count

def compression_ratio_evaluation(preprocessed_file, summary):
    """
    Run compression ratio evaluation on a summary.
    """
    preprocessed_sentences = load_preprocessed_json(preprocessed_file)
    compression_ratio = compute_compression_ratio(preprocessed_sentences, summary)

    return {"Compression Ratio": round(compression_ratio, 2)}
