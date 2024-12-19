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

def summarize_lecture_chunks(lecture_chunks):
    """
    Summarize a list of text chunks for a lecture and combine the summaries.
    """
    chunk_summaries = []
    for chunk in lecture_chunks:
        print("Summarizing chunk...")
        text_chunks = split_into_chunks(chunk)
        for text_chunk in text_chunks:
            chunk_summary = summarize_text_distilbart(text_chunk)
            chunk_summaries.append(chunk_summary)

    combined_summary = " ".join(chunk_summaries)

    # If the combined summary is too long, summarize it again
    if len(tokenizer.tokenize(combined_summary)) > 1024:
        final_summary = summarize_text_distilbart(combined_summary)
    else:
        final_summary = combined_summary

    return final_summary

def process_and_save_summaries(data_paths, output_dir="summarized_lectures_distilbart"):
    """
    Process all JSON files in the data_paths, summarize lectures, and save results.
    """
    os.makedirs(output_dir, exist_ok=True)

    for method, directory in data_paths.items():
        print(f"Processing directory: {directory}...")
        data = load_preprocessed_data(directory)

        # Create a subdirectory for each method
        method_output_dir = os.path.join(output_dir, method)
        os.makedirs(method_output_dir, exist_ok=True)

        for lecture_name, chunks in data.items():
            print(f"Summarizing lecture: {lecture_name}...")
            lecture_summary = summarize_lecture_chunks(chunks)

            # Save the summary as a text file
            lecture_file_path = os.path.join(method_output_dir, f"{lecture_name}.txt")
            with open(lecture_file_path, "w", encoding="utf-8") as f:
                f.write(lecture_summary)

            print(f"Saved summary for lecture: {lecture_name} to {lecture_file_path}")

# Run the summarization process
process_and_save_summaries(data_paths)
