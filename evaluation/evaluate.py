import os
import json
from semantic import evaluate_semantic
from compression import evaluate_compression

ORIGINALS_DIR = "dataset/preprocessed/C"
SUMMARIES_DIR = "dataset/summarized_lectures_bart/C"
RESULTS_FILE = "evaluation/results.txt"

def load_original_text(filename: str) -> str:
    """Load and join text from JSON file."""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Join all sentences in the array
    return " ".join(data)

def load_summary_text(filename: str) -> str:
    """Load summary text from file."""
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().strip()

def main():
    # Create evaluation directory if it doesn't exist
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    
    # Gather all originals and their matching summaries
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

            # Calculate metrics
            semantic_scores = evaluate_semantic(original_text, summary_text)
            compression_scores = evaluate_compression(original_text, summary_text)

            # Combine results
            combined_result = {
                "file": base_name,
                **semantic_scores,
                **compression_scores
            }
            results.append(combined_result)
            print(f"Processed {base_name}")
            
        except Exception as e:
            print(f"Error processing {orig_file}: {str(e)}")
            continue

    # Write results to file
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        # Updated header to include all metrics
        f.write("file,document_similarity,sentence_similarity,compression_score\n")
        for r in results:
            f.write(f"{r['file']},{r['document_similarity']:.4f},{r['sentence_similarity']:.4f},{r['compression_score']:.4f}\n")

    # Print average scores
    if results:
        avg_doc_sim = sum(r['document_similarity'] for r in results) / len(results)
        avg_sent_sim = sum(r['sentence_similarity'] for r in results) / len(results)
        avg_compression = sum(r['compression_score'] for r in results) / len(results)
        print(f"\nAverage Scores across {len(results)} lectures:")
        print(f"Document Similarity: {avg_doc_sim:.4f}")
        print(f"Sentence Similarity: {avg_sent_sim:.4f}")
        print(f"Compression Score: {avg_compression:.4f}")

if __name__ == "__main__":
    main()