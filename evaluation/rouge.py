from rouge_score import rouge_scorer
import pandas as pd
import os
import glob

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

original_dir = "../dataset/unprocessed/"
summary_dir = "../dataset/summarized_lectures_bart/C/"

metrics = ['rouge1', 'rouge2', 'rougeL']
precision_sum = {metric: 0 for metric in metrics}
recall_sum = {metric: 0 for metric in metrics}
f1_sum = {metric: 0 for metric in metrics}
file_count = 0

for i in range(1, 23):
    original_file = os.path.join(original_dir, f"lec{i}.txt")
    summary_file = os.path.join(summary_dir, f"lec{i}.txt")

    if not os.path.exists(original_file) or not os.path.exists(summary_file):
        print(f"Skipping lec{i}.txt: Missing file.")
        continue

    with open(original_file, "r", encoding="utf-8") as f:
        original_text = f.read()
    with open(summary_file, "r", encoding="utf-8") as f:
        summary_text = f.read()

    # Calculate ROUGE scores
    scores = scorer.score(original_text, summary_text)

    # Accumulate scores
    for metric in metrics:
        precision_sum[metric] += scores[metric].precision
        recall_sum[metric] += scores[metric].recall
        f1_sum[metric] += scores[metric].fmeasure

    file_count += 1

# Calculate averages
average_scores = {
    "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
    "Average Precision": [precision_sum[metric] / file_count for metric in metrics],
    "Average Recall": [recall_sum[metric] / file_count for metric in metrics],
    "Average F1 Score": [f1_sum[metric] / file_count for metric in metrics]
}

df = pd.DataFrame(average_scores)

df.to_csv("average_rouge_scores.csv", index=False)
