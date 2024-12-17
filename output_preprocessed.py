import os
import json
from final_pipelines import (
    pipeline_sentence_level,
    pipeline_fixed_chunks,
    pipeline_topic_based
)

def save_output(directory, filename, data):
    """ Save data to JSON file. """
    os.makedirs(directory, exist_ok=True)
    output_path = os.path.join(directory, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Saved: {output_path}")

def main():
    input_dir = "dataset/unprocessed"
    output_dir = "dataset/preprocessed"

    for lecture_file in os.listdir(input_dir):
        if lecture_file.endswith(".txt"):
            file_path = os.path.join(input_dir, lecture_file)
            lecture_name = os.path.splitext(lecture_file)[0]

            print(f"Processing {lecture_file}...")

            # Pipeline A: Sentence-level splitting
            sentences = pipeline_sentence_level(file_path)
            save_output(os.path.join(output_dir, "A"), f"{lecture_name}.json", sentences)

            # Pipeline B: Fixed sentence chunking (5, 10, 30)
            for size in [5, 10, 30]:
                chunks = pipeline_fixed_chunks(file_path, size)
                save_output(os.path.join(output_dir, "B", str(size)), f"{lecture_name}.json", chunks)

            # Pipeline C: Topic-based grouping
            topics = pipeline_topic_based(file_path, n_topics=5) # 5 topics per lecture 
            save_output(os.path.join(output_dir, "C"), f"{lecture_name}.json", topics)

    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()
