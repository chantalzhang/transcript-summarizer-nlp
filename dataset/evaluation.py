from rouge_score import rouge_scorer

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Path to the preprocessed file
preprocessed_file = "./unprocessed/lec12.txt"

# Read the content of the preprocessed file
with open(preprocessed_file, "r", encoding="utf-8") as f:
    original_text = f.read()

# Path to the preprocessed file
summary_file = "./summarized_lectures_bart/B/lec12.txt"

# Read the content of the preprocessed file
with open(summary_file, "r", encoding="utf-8") as f:
   summary_text = f.read()



# Calculate ROUGE scores
scores = scorer.score(original_text, summary_text)

# Print the scores
print("ROUGE-1:", scores['rouge1'])
print("ROUGE-2:", scores['rouge2'])
print("ROUGE-L:", scores['rougeL'])