from pipeline import pipeline_a, pipeline_b, pipeline_c, pipeline_d, pipeline_e
from evaluation import calculate_precision_recall_f1, calculate_cosine_similarity
from generate_reference import generate_reference_keywords

def evaluate_pipelines(file_path, reference_keywords):
    pipelines = {
        "Pipeline A": pipeline_a,
        "Pipeline B": pipeline_b,
        "Pipeline C": pipeline_c,
        "Pipeline D": pipeline_d,
        "Pipeline E": pipeline_e,
    }

    for name, pipeline_func in pipelines.items():
        print(f"\n### Evaluating {name} ###")
        
        result = pipeline_func(file_path)

        extracted_keywords = result.get("keywords", [])
        if not extracted_keywords:
            cleaned_text = " ".join(result.get("simplified_sentences", []))
            extracted_keywords = extract_keywords_tfidf(cleaned_text, num_keywords=10)

        precision, recall, f1_score = calculate_precision_recall_f1(extracted_keywords, reference_keywords)
        print("\nKeyword Evaluation:")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f}")

        topics = result.get("topics", {})
        if topics:
            print("\nCluster Evaluation:")
            for topic, sentences in topics.items():
                avg_similarity = calculate_cosine_similarity(sentences)
                print(f"{topic} - Average Cosine Similarity: {avg_similarity:.2f}")

if __name__ == "__main__":
    lecture_num = input("Enter lecture number (1-22): ")
    file_path = f"dataset/mycourses/lec{lecture_num}.txt"

    reference_keywords = generate_reference_keywords(file_path, num_keywords=20)
    print("\nGenerated Reference Keywords:")
    print(reference_keywords)

    evaluate_pipelines(file_path, reference_keywords)
