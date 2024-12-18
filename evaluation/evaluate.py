import os
import json
from semantic import evaluate_semantic
from compression import evaluate_compression
from rouge import evaluate_rouge

ORIGINALS_DIR = "dataset/preprocessed/C"
SUMMARIES_DIR = "dataset/summarized_lectures_bart/C"
RESULTS_FILE = "evaluation/results.txt"

def load_original_text(filename: str) -> str:
    """Load and join text from JSON file."""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return " ".join(data)

def load_summary_text(filename: str) -> str:
    """Load summary text from file."""
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().strip()

def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    originals = [f for f in os.listdir(ORIGINALS_DIR) if f.endswith(".json")]
    results = []

    for orig_file in originals:
        base_name = os.path.splitext(orig_file)[0]
        summary_file = base_name + ".txt"

        original_path = os.path.join(ORIGINALS_DIR, orig_file)
        summary_path = os.path.join(SUMMARIES_DIR, summary_file)

        if not os.path.exists(summary_path):
            print(f"No summary found for {orig_file}, skipping.")
            continue

        try:
            original_text = load_original_text(original_path)
            summary_text = load_summary_text(summary_path)

            # Calculate all metrics
            semantic_scores = evaluate_semantic(original_text, summary_text)
            compression_scores = evaluate_compression(original_text, summary_text)
            rouge_scores = evaluate_rouge(original_text, summary_text)

            # Combine results
            combined_result = {
                "file": base_name,
                **semantic_scores,
                **compression_scores,
                **rouge_scores
            }
            results.append(combined_result)
            print(f"Processed {base_name}")
            
        except Exception as e:
            print(f"Error processing {orig_file}: {str(e)}")
            continue

    # Write results to file
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        # Updated header to include ROUGE scores
        headers = ["file", "document_similarity", "sentence_similarity", 
                  "compression_score", "rouge_1_precision", "rouge_1_recall", 
                  "rouge_1_f1", "rouge_2_precision", "rouge_2_recall", "rouge_2_f1"]
        f.write(",".join(headers) + "\n")
        
        for r in results:
            values = [str(r.get(h, "")) if h != "file" else r[h] for h in headers]
            f.write(",".join(values) + "\n")

    # Print average scores
    if results:
        print(f"\nAverage Scores across {len(results)} lectures:")
        metrics = [
            ("Document Similarity", "document_similarity"),
            ("Sentence Similarity", "sentence_similarity"),
            ("Compression Score", "compression_score"),
            ("ROUGE-1 F1", "rouge_1_f1"),
            ("ROUGE-2 F1", "rouge_2_f1")
        ]
        
        for metric_name, metric_key in metrics:
            avg_score = sum(r[metric_key] for r in results) / len(results)
            print(f"{metric_name}: {avg_score:.4f}")

if __name__ == "__main__":
    main()