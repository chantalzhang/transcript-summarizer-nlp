# this is not thta good for our task 


# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# from rake_nltk import Rake
# from keybert import KeyBERT
# import logging

# # Suppress loading bars and logs
# logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# def calculate_precision_recall_f1(extracted, reference):
#     extracted_set = set(extracted)
#     reference_set = set(reference)
#     true_positives = len(extracted_set & reference_set)
#     false_positives = len(extracted_set - reference_set)
#     false_negatives = len(reference_set - extracted_set)
    
#     precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
#     recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
#     f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
#     return precision, recall, f1_score

# def calculate_cosine_similarity(sentences):
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     embeddings = model.encode(sentences)
#     similarity_matrix = cosine_similarity(embeddings)
#     avg_similarity = np.mean(similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)])
#     return avg_similarity

# def extract_keywords_tfidf(text, num_keywords=10):
#     if not text.strip():
#         return []  # Handle empty text gracefully
#     vectorizer = TfidfVectorizer(stop_words='english', max_features=num_keywords)
#     tfidf_matrix = vectorizer.fit_transform([text])
#     feature_names = vectorizer.get_feature_names_out()
#     return list(feature_names)

# def generate_reference_keywords(text, method="rake", num_keywords=10):
#     if not text.strip():
#         return []  # Handle empty text gracefully
#     if method == "rake":
#         rake = Rake()
#         rake.extract_keywords_from_text(text)
#         return rake.get_ranked_phrases()[:num_keywords]
#     elif method == "keybert":
#         kw_model = KeyBERT('all-MiniLM-L6-v2')
#         return [kw[0] for kw in kw_model.extract_keywords(text, top_n=num_keywords)]
#     else:
#         raise ValueError("Unsupported method for keyword generation. Use 'rake' or 'keybert'.")

# if __name__ == "__main__":
#     lecture_num = input("Enter lecture number (1-22): ")
#     file_path = f"dataset/mycourses/lec{lecture_num}.txt"

#     from pipeline import pipeline_a, pipeline_b, pipeline_c, pipeline_d, pipeline_e

#     pipelines = {
#         "Pipeline A": pipeline_a,
#         "Pipeline B": pipeline_b,
#         "Pipeline C": pipeline_c,
#         "Pipeline D": pipeline_d,
#         "Pipeline E": pipeline_e
#     }

#     for name, pipeline_func in pipelines.items():
#         print(f"\n### Evaluating {name} ###")
#         pipeline_result = pipeline_func(file_path)

#         cleaned_text = " ".join(pipeline_result.get("simplified_sentences", []))
#         if not cleaned_text.strip():
#             print("Warning: Cleaned text is empty. Skipping evaluation for this pipeline.")
#             continue

#         # Extract keywords from cleaned text
#         extracted_keywords = extract_keywords_tfidf(cleaned_text, num_keywords=10)

#         # Generate reference keywords automatically
#         reference_keywords = generate_reference_keywords(cleaned_text, method="rake", num_keywords=10)

#         if not reference_keywords:
#             print("Warning: Reference keywords are empty. Skipping evaluation for this pipeline.")
#             continue

#         # Keyword Evaluation
#         precision, recall, f1_score = calculate_precision_recall_f1(extracted_keywords, reference_keywords)
#         print("\nKeyword Evaluation:")
#         print(f"Precision: {precision:.2f}")
#         print(f"Recall: {recall:.2f}")
#         print(f"F1-Score: {f1_score:.2f}")

#         # Cluster Evaluation
#         print("\nCluster Evaluation:")
#         for topic, sentences in pipeline_result.get("topics", {}).items():
#             avg_similarity = calculate_cosine_similarity(sentences)
#             print(f"{topic} - Average Cosine Similarity: {avg_similarity:.2f}")
