# import weaviate as wc
# import weaviate.classes as wvc
# import os
# from dotenv import load_dotenv
# import os
# from transformers import pipeline
# from nltk.tokenize import sent_tokenize
# import nltk
# import re
# import unicodedata
# import warnings

# # nltk.download('punkt')  # Download sentence tokenizer

# # abstractive summarization model
# summarizer = pipeline("summarization", model="google/pegasus-xsum")

# # Suppress ResourceWarning for temporary files
# warnings.filterwarnings("ignore", category=ResourceWarning)

# load_dotenv()

# wcd_url = os.getenv("WCD_URL")
# wcd_api_key = os.getenv("WCD_API_KEY")
# huggingface_key = os.getenv("HF_KEY")
# headers = {
#     "X-HuggingFace-Api-Key": huggingface_key,
# }

# # Connect to a WCS instance
# client = wc.connect_to_weaviate_cloud(
#     cluster_url=wcd_url,                             
#     auth_credentials=wvc.init.Auth.api_key(wcd_api_key),  
#     headers=headers
# )

# def normalize_sentence(sentence):
#     """
#     Normalize a sentence for comparison:
#     - Normalize Unicode characters.
#     - Strip leading/trailing whitespace.
#     - Replace multiple spaces with a single space.
#     - Convert to lowercase.
#     """
#     # Normalize Unicode to NFC (Canonical Composition)
#     sentence = unicodedata.normalize('NFC', sentence)
#     # Strip leading/trailing whitespace
#     sentence = sentence.strip()
#     # Convert to lowercase
#     sentence = sentence.lower()
#     # Replace multiple spaces with a single space
#     sentence = re.sub(r'\s+', ' ', sentence)
#     return sentence

# def retrieve_relevant_sentences(client, collection_name, query_text, num_results=5):
#     """
#     Retrieve contextually relevant sentences from Weaviate.
#     """
#     collection = client.collections.get(collection_name)

#     response = collection.query.near_text(
#         query=query_text,
#         distance=0.8,   # max accepted distance (default distance metric is cosine)
#         limit=num_results
#     )

#     sentences = []
#     normalized_query = normalize_sentence(query_text)
#     seen_sentences = set(normalized_query)

#     for obj in response.objects:
#         sentence = obj.properties["sentence"]
#         normalized = normalize_sentence(sentence)
#         if normalized not in seen_sentences:  # Only add unique sentences
#             sentences.append(normalized)
#             seen_sentences.add(normalized)

#     return sentences

# def aggregate_context(current_lecture_text, retrieved_sentences):
#     """
#     Combine the current lecture text with retrieved sentences.
#     """
#     combined_text = current_lecture_text + "\n" + "\n".join(retrieved_sentences)
#     return combined_text

# def split_into_chunks(text, max_words=500):
#     """
#     Split text into chunks of approximately max_words words.
#     """
#     sentences = sent_tokenize(text)
#     chunks = []
#     current_chunk = []
#     current_word_count = 0

#     for sentence in sentences:
#         word_count = len(sentence.split())
#         if current_word_count + word_count > max_words:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = []
#             current_word_count = 0
#         current_chunk.append(sentence)
#         current_word_count += word_count

#     # Add the last chunk
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
    
#     return chunks

# def process_lecture(file_path, collection_name):
#     """
#     Split the lecture into sentences, retrieve similar sentences, and generate summaries.
#     """
#     with open(file_path, "r", encoding="utf-8") as f:
#         lecture_text = f.read()
    
#     sentences = sent_tokenize(lecture_text)
#     print(f"Total sentences in lecture: {len(sentences)}")
    
#     enriched_sentences = []
#     for i, sentence in enumerate(sentences):
#         print(f"Processing sentence {i + 1}/{len(sentences)}: {sentence}")
        
#         # Retrieve similar sentences from Weaviate
#         similar_sentences = retrieve_relevant_sentences(client, collection_name, sentence, num_results=3)
        
#         # Concatenate the current sentence with retrieved similar sentences
#         combined_text = sentence + " " + " ".join(similar_sentences)
#         print(combined_text)
#         enriched_sentences.append(combined_text)
    
#     combined_enriched_text = " ".join(enriched_sentences)
    
#     # Generate the final summary
#     print("Generating final summary...")
#     final_summary = generate_abstractive_summary(combined_enriched_text)
    
#     return final_summary

# def save_to_file(content, output_dir, filename):
#     """
#     Save content to a file in the specified directory.

#     Args:
#         content (str): The text content to save.
#         output_dir (str): The directory where the file will be saved.
#         filename (str): The name of the output file.
#     """
#     os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
#     file_path = os.path.join(output_dir, filename)
#     with open(file_path, "w", encoding="utf-8") as file:
#         file.write(content)
#     print(f"Saved to {file_path}")

# def summarize_chunks(chunks, max_length=150, min_length=50):
#     """
#     Summarize each chunk using the pre-trained summarization model.
#     """
#     summaries = []
#     for i, chunk in enumerate(chunks):
#         print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
#         summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
#         summaries.append(summary[0]["summary_text"])
#     return summaries

# def summarize_chunks_hierarchically(chunks, summarizer, max_length=150, min_length=50):
#     """
#     Summarize each chunk and return intermediate summaries.
#     """
#     summaries = []
#     for i, chunk in enumerate(chunks):
#         print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
#         summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
#         summaries.append(summary[0]["summary_text"])
#     return summaries

# def summarize_in_levels(summaries, summarizer, max_words_per_group=500, max_length=500, min_length=100):
#     """
#     Summarize intermediate summaries hierarchically if they exceed the token limit.
#     """
#     # Step 1: Split summaries into groups
#     grouped_summaries = split_into_chunks(" ".join(summaries), max_words=max_words_per_group)
    
#     # Step 2: Summarize each group
#     refined_summaries = summarize_chunks_hierarchically(grouped_summaries, summarizer, max_length, min_length)
    
#     return refined_summaries

# def hierarchical_summarization_pipeline(lecture_text, summarizer, output_dir, filename_prefix, max_words_per_chunk=500, max_words_per_group=500):
#     """
#     Full pipeline to summarize long lectures hierarchically.
#     """
#     # Step 1: Split the lecture into chunks
#     chunks = split_into_chunks(lecture_text, max_words=max_words_per_chunk)
#     print(f"Initial chunks created: {len(chunks)}")
    
#     # Step 2: Summarize first-level chunks
#     chunk_summaries = summarize_chunks_hierarchically(chunks, summarizer)
    
#     # Step 3: Check if intermediate summaries are too long
#     combined_summaries = " ".join(chunk_summaries)
#     if len(combined_summaries.split()) > max_words_per_group:
#         print("Intermediate summaries are too long, summarizing again...")
#         refined_summaries = summarize_in_levels(chunk_summaries, summarizer, max_words_per_group)
#     else:
#         refined_summaries = chunk_summaries
    
#     # Step 4: Generate the final summary
#     final_summary = summarize_in_levels(refined_summaries, summarizer, max_words_per_group)
#     final_summary_text = "\n".join(final_summary)
#     save_to_file(final_summary_text, output_dir, f"{filename_prefix}_final_summary.txt")
#     return final_summary

# if __name__ == "__main__":
    # with open("../dataset/mycourses/lec1.txt", "r", encoding="utf-8") as file:
    #     lecture_text = file.read()

    # output_directory = "raw_summaries"
    # filename_prefix = "lec1"
    
    # # Run the hierarchical summarization pipeline
    # final_summary = hierarchical_summarization_pipeline(
    #     lecture_text, summarizer, output_directory, filename_prefix, max_words_per_chunk=500, max_words_per_group=300
    # )
    
    # print("\nFinal Summary:")
    # print(" ".join(final_summary))
