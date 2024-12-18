import os
import json
from coverage import evaluate_coverage
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
            coverage_scores = evaluate_coverage(original_text, summary_text)
            compression_scores = evaluate_compression(original_text, summary_text)

            # Combine results
            combined_result = {
                "file": base_name,
                **coverage_scores,
                **compression_scores
            }
            results.append(combined_result)
            print(f"Processed {base_name}")
            
        except Exception as e:
            print(f"Error processing {orig_file}: {str(e)}")
            continue

    # Write results to file
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("file,keyphrase_coverage_score,compression_score\n")
        for r in results:
            f.write(f"{r['file']},{r['keyphrase_coverage_score']:.4f},{r['compression_score']:.4f}\n")

    # Print average scores
    if results:
        avg_coverage = sum(r['keyphrase_coverage_score'] for r in results) / len(results)
        avg_compression = sum(r['compression_score'] for r in results) / len(results)
        print(f"\nAverage Scores across {len(results)} lectures:")
        print(f"Coverage Score: {avg_coverage:.4f}")
        print(f"Compression Score: {avg_compression:.4f}")

if __name__ == "__main__":
    main()