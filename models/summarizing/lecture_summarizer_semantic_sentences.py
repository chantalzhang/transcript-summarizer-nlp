import os
import json
import weaviate as wc
import weaviate.classes as wvc
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoModel, AutoTokenizer
import torch
from dotenv import load_dotenv

load_dotenv()

# model_name = "facebook/bart-large-cnn"
# tokenizer = BartTokenizer.from_pretrained(model_name)
# # model = BartForConditionalGeneration.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
# model = AutoModel.from_pretrained("EleutherAI/gpt-neo", cache_dir="~/.cache/huggingface/hub")
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

WCD_URL = os.getenv("WCD_URL")
WCD_API_KEY = os.getenv("WCD_API_KEY")
HF_KEY = os.getenv("HF_KEY")
headers = {
    "X-HuggingFace-Api-Key": HF_KEY,
}

client = wc.connect_to_weaviate_cloud(
    cluster_url=WCD_URL,
    auth_credentials=wvc.init.Auth.api_key(WCD_API_KEY),
    headers=headers
)

def load_lecture_sentences(file_path):
    """Load sentences from a JSON lecture file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_similar_sentences(sentence, current_lecture, current_id, distance=0.5, top_k=2):
    """
    Use Weaviate's semantic search to find similar sentences in the database.
    Ensure results are not from the same lecture or sentence ID.
    """
    near_text = {
        "concepts": [sentence],
        "distance": distance
    }

    response = client.query.get("Sentences", ["sentence", "lecture_name", "id", "_additional {distance}"]).with_near_text(near_text).with_limit(top_k).do()

    results = []
    if "data" in response and "Get" in response["data"] and "Sentences" in response["data"]["Get"]:
        for result in response["data"]["Get"]["Sentences"]:
            if (
                result["lecture_name"] != current_lecture
                and result["id"] != current_id
                and result["sentence"] != sentence
                and result["_additional"]["distance"] <= distance
            ):
                results.append(result["sentence"])
    return results

def split_into_chunks(text, max_tokens=1024):
    """
    Split the input text into chunks of a specified maximum number of tokens.
    """
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(tokenizer.tokenize(" ".join(current_chunk))) >= max_tokens:
            current_chunk.pop()
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def summarize_text_bart(text, max_length=150):
    """
    Summarize text using BART model.
    """
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_lecture_sentences(file_path, output_path):
    """
    Summarize lecture sentences with semantic chunks from Weaviate.
    """
    lecture_data = load_lecture_sentences(file_path)
    lecture_name = os.path.basename(file_path).split(".")[0]
    all_chunks = []

    for sentence_data in lecture_data:
        sentence = sentence_data["sentence"]
        sentence_id = sentence_data["id"]
        similar_sentences = find_similar_sentences(sentence, lecture_name, sentence_id)

        # Combine current sentence and similar sentences into a chunk
        chunk = sentence + " " + " ".join(similar_sentences)
        all_chunks.append(chunk)

    # Summarize the collected chunks
    chunk_summaries = []
    for chunk in all_chunks:
        text_chunks = split_into_chunks(chunk)
        for text_chunk in text_chunks:
            chunk_summary = summarize_text_bart(text_chunk)
            chunk_summaries.append(chunk_summary)

    combined_summary = " ".join(chunk_summaries)

    # If combined summary is too long, summarize again
    if len(tokenizer.tokenize(combined_summary)) > 1024:
        final_summary = summarize_text_bart(combined_summary)
    else:
        final_summary = combined_summary

    # Save the final summary
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_summary)

    print(f"Summary saved to {output_path}")

lecture_file = "../../dataset/preprocessed/A/lec12.json"
output_file = "summarized_lectures/lec12_summary.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
summarize_lecture_sentences(lecture_file, output_file)

client.close()
