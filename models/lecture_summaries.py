# import os
# import json
# import openai
# from dotenv import load_dotenv

# # Load API key from .env file
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_KEY")

# # Directory paths
# data_paths = {
#     "B": "../dataset/preprocessed/B/30",  # Chunk-level
#     "C": "../dataset/preprocessed/C",  # Topic-level
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

# def summarize_with_openai(text, max_tokens=8192):
#     """
#     Summarize a text using OpenAI API.
#     """
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are an assistant that summarizes lecture transcripts."},
#                 {"role": "user", "content": f"Summarize the following text:\n{text}"}
#             ],
#             max_tokens=max_tokens,
#             temperature=0.3,
#         )
#         return response['choices'][0]['message']['content'].strip()
#     except Exception as e:
#         print(f"Error during OpenAI summarization: {e}")
#         return "Summary generation failed."

# def summarize_lecture_chunks(lecture_chunks):
#     """
#     Summarize a list of text chunks for a lecture and combine the summaries.
#     """
#     chunk_summaries = []
#     for chunk in lecture_chunks:
#         print("Summarizing chunk...")
#         chunk_summary = summarize_with_openai(chunk, max_tokens=1024)
#         chunk_summaries.append(chunk_summary)

#     # Combine all chunk summaries into a final summary
#     combined_summary = " ".join(chunk_summaries)
#     final_summary = summarize_with_openai(combined_summary, max_tokens=1024)
#     return final_summary

# def process_and_save_summaries(data_paths, output_dir="summarized_lectures"):
#     """
#     Process all JSON files in the data_paths, summarize lectures, and save results.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     for method, directory in data_paths.items():
#         print(f"Processing directory: {directory}...")
#         data = load_preprocessed_data(directory)

#         # Create a subdirectory for each method
#         method_output_dir = os.path.join(output_dir, method)
#         os.makedirs(method_output_dir, exist_ok=True)

#         for lecture_name, chunks in data.items():
#             print(f"Summarizing lecture: {lecture_name}...")
#             lecture_summary = summarize_lecture_chunks(chunks)

#             # Save the summary as a text file
#             lecture_file_path = os.path.join(method_output_dir, f"{lecture_name}.txt")
#             with open(lecture_file_path, "w", encoding="utf-8") as f:
#                 f.write(lecture_summary)

#             print(f"Saved summary for lecture: {lecture_name} to {lecture_file_path}")

# # Run the summarization process
# process_and_save_summaries(data_paths)

import os
import json
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

model_name = "Feluda/pegasus-samsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Directory paths
data_paths = {
    "B": "../dataset/preprocessed/B/30",  # Chunk-level
    "C": "../dataset/preprocessed/C",     # Topic-level
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

def summarize_with_pegasus(text, max_length=5000, min_length=1000):
    """
    Summarize text using the Pegasus model.
    """
    inputs = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=10,
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
        chunk_summary = summarize_with_pegasus(chunk)
        chunk_summaries.append(chunk_summary)

    # Combine all chunk summaries into a final summary
    combined_summary = " ".join(chunk_summaries)
    final_summary = summarize_with_pegasus(combined_summary)
    return final_summary

def process_and_save_summaries(data_paths, output_dir="summarized_lectures"):
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
process_and_save_summaries(data_paths, "summarized_lectures_samsum")
