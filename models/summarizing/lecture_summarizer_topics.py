# import os
# import json
# from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

# # model_name = "facebook/bart-large-cnn"
# # tokenizer = BartTokenizer.from_pretrained(model_name)
# # model = BartForConditionalGeneration.from_pretrained(model_name)
# model_name = "sshleifer/distilbart-cnn-12-6"
# tokenizer = BartTokenizer.from_pretrained(model_name)
# model = BartForConditionalGeneration.from_pretrained(model_name)

# # Directory paths
# data_paths = {
#     "C": "../dataset/preprocessed/C",     # Topic-level
# }

# def load_preprocessed_data(directory):
#     """Load all JSON files in a directory."""
#     data = {}
#     for file_name in sorted(os.listdir(directory)):
#         if file_name.endswith(".json"):
#             with open(os.path.join(directory, file_name), "r", encoding="utf-8") as f:
#                 lecture_name = file_name.split(".")[0]
#                 data[lecture_name] = json.load(f)
#     return data

# def split_into_chunks(text, max_tokens=1024):
#     """Split the input text into chunks."""
#     words = text.split()
#     chunks = []
#     current_chunk = []

#     for word in words:
#         current_chunk.append(word)
#         if len(tokenizer.tokenize(' '.join(current_chunk))) >= max_tokens:
#             current_chunk.pop()
#             chunks.append(' '.join(current_chunk))
#             current_chunk = [word]

#     if current_chunk:
#         chunks.append(' '.join(current_chunk))

#     return chunks

# def summarize_with_bart(text, max_length=50, min_length=50):
#     """Summarize text using the BART model."""
#     inputs = tokenizer(
#         text,
#         truncation=True,
#         max_length=1024,
#         padding="longest",
#         return_tensors="pt"
#     )
#     summary_ids = model.generate(
#         inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         max_length=max_length,
#         min_length=min_length,
#         length_penalty=2.0,
#         num_beams=4,
#         early_stopping=True
#     )
#     return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# def summarize_topic(topic_data):
#     """Summarize all chunks within a topic."""
#     combined_summary = []
#     for chunk in topic_data:
#         print("Summarizing chunk...")
#         chunk_chunks = split_into_chunks(chunk)
#         for sub_chunk in chunk_chunks:
#             summary = summarize_with_bart(sub_chunk)
#             combined_summary.append(summary)
#     return " ".join(combined_summary)

# def summarize_lecture_by_topics(lecture_data):
#     """Summarize lecture data organized by topics."""
#     topic_summaries = {}
#     for topic, chunks in lecture_data.items():
#         print(f"Summarizing topic: {topic}...")
#         topic_summary = summarize_topic(chunks)
#         topic_summaries[topic] = topic_summary
#     return topic_summaries

# def save_summaries_by_topic(summaries, output_dir, lecture_name):
#     """Save summaries by topic to text files."""
#     lecture_dir = os.path.join(output_dir, lecture_name)
#     os.makedirs(lecture_dir, exist_ok=True)

#     for topic, summary in summaries.items():
#         topic_file_path = os.path.join(lecture_dir, f"{topic}.txt")
#         with open(topic_file_path, "w", encoding="utf-8") as f:
#             f.write(summary)
#         print(f"Saved summary for topic: {topic} to {topic_file_path}")

# def process_and_save_summaries(data_paths, output_dir="summarized_lectures"):
#     """Process all JSON files, summarize by topic, and save results."""
#     os.makedirs(output_dir, exist_ok=True)

#     for method, directory in data_paths.items():
#         print(f"Processing directory: {directory}...")
#         data = load_preprocessed_data(directory)

#         method_output_dir = os.path.join(output_dir, method)
#         os.makedirs(method_output_dir, exist_ok=True)

#         for lecture_name, lecture_data in data.items():
#             print(f"Summarizing lecture: {lecture_name}...")
#             topic_summaries = summarize_lecture_by_topics(lecture_data)
#             save_summaries_by_topic(topic_summaries, method_output_dir, lecture_name)

# # Run the summarization process
# process_and_save_summaries(data_paths, "summarized_lectures_distilbart")

import os
import json
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from dotenv import load_dotenv

load_dotenv()

# Load DistilBART model and tokenizer
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
data_paths = {
    "C": "../../dataset/preprocessed/C",  # Chunk-level
}

def load_preprocessed_data(directory):
    """Load all JSON files in a directory."""
    data = {}
    for file_name in sorted(os.listdir(directory)):
        if file_name.endswith(".json"):
            with open(os.path.join(directory, file_name), "r", encoding="utf-8") as f:
                lecture_name = file_name.split(".")[0]
                data[lecture_name] = json.load(f)
    return data

def split_into_chunks(text, max_tokens=1024):
    """
    Split the input text into chunks of a specified maximum number of tokens.
    """
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        # Check if the current chunk exceeds the max_tokens limit
        if len(tokenizer.tokenize(' '.join(current_chunk))) >= max_tokens:
            # Remove the last word to fit within the limit
            current_chunk.pop()
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]

    # Add any remaining words as the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def summarize_text_distilbart(text, max_length=150):
    """
    Summarize text using the DistilBART model.
    """
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_topics(lecture_data):
    """
    Summarize each topic within a lecture.
    """
    topic_summaries = {}
    for topic, content in lecture_data.items():
        print(f"Summarizing {topic}...")
        topic_text = " ".join(content)
        text_chunks = split_into_chunks(topic_text)
        chunk_summaries = []
        for text_chunk in text_chunks:
            chunk_summary = summarize_text_distilbart(text_chunk)
            chunk_summaries.append(chunk_summary)
        topic_summary = " ".join(chunk_summaries)

        # If the topic summary is too long, summarize it again
        if len(tokenizer.tokenize(topic_summary)) > 1024:
            topic_summary = summarize_text_distilbart(topic_summary)

        topic_summaries[topic] = topic_summary
    return topic_summaries

def process_and_save_summaries_by_topic(data_paths, output_dir="summarized_lectures_by_topic"):
    """
    Process all JSON files in the data_paths, summarize topics within lectures, and save results.
    """
    os.makedirs(output_dir, exist_ok=True)

    for method, directory in data_paths.items():
        print(f"Processing directory: {directory}...")
        data = load_preprocessed_data(directory)

        # Create a subdirectory for each method
        method_output_dir = os.path.join(output_dir, method)
        os.makedirs(method_output_dir, exist_ok=True)

        for lecture_name, lecture_data in data.items():
            print(f"Summarizing lecture: {lecture_name} by topics...")
            topic_summaries = summarize_topics(lecture_data)

            # Save the topic summaries as a JSON file
            lecture_file_path = os.path.join(method_output_dir, f"{lecture_name}_topics.json")
            with open(lecture_file_path, "w", encoding="utf-8") as f:
                json.dump(topic_summaries, f, indent=4)

            print(f"Saved topic summaries for lecture: {lecture_name} to {lecture_file_path}")

process_and_save_summaries_by_topic(data_paths)
