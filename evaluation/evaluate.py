from finesure import finesure_evaluation
from compression import compression_ratio_evaluation
from coverage import coverage_score_evaluation

def full_evaluation(preprocessed_file, summary):
    """
    Run FineSurE, Compression Ratio, and Coverage Score evaluations on a summary.
    """
    finesure_results = finesure_evaluation(preprocessed_file, summary)
    compression_results = compression_ratio_evaluation(preprocessed_file, summary)
    coverage_results = coverage_score_evaluation(preprocessed_file, summary)

    # Combine results
    evaluation_results = {**finesure_results, **compression_results, **coverage_results}
    return evaluation_results

# Example Usage
if __name__ == "__main__":
    preprocessed_file = "dataset/preprocessed/B/lec9.json"
    summary = "dataset/summarized_lectures_bart/B/lec9.text"
    
    results = full_evaluation(preprocessed_file, summary)
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")
