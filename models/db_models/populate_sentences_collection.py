import os
import weaviate
import weaviate.classes as wvc
from weaviate.util import generate_uuid5 
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
import nltk
import json

# nltk tokenizer data
nltk.download('punkt')

load_dotenv()

DATA_DIR = "../../dataset/preprocessed/A"
wcd_url = os.getenv("WCD_URL")
wcd_api_key = os.getenv("WCD_API_KEY")
huggingface_key = os.getenv("HF_KEY")
headers = {
    "X-HuggingFace-Api-Key": huggingface_key,
}

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,
    auth_credentials=wvc.init.Auth.api_key(wcd_api_key),
    headers=headers
)

def process_and_insert_sentences_from_json_array(json_dir, collection_name):
    """
    Process JSON files containing arrays of sentences and insert them into the Weaviate collection.
    """
    for file_name in os.listdir(json_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(json_dir, file_name)
            lecture_name = file_name.replace(".json", "")
            print(f"Processing {file_path}...")

            with open(file_path, "r", encoding="utf-8") as f:
                sentences = json.load(f) # json file is a list of sentences

            # prepare data objects for batch insertion
            data_objects = []
            for sentence_id, sentence in enumerate(sentences):
                data_object = {
                    "sentence": sentence,
                    "lecture_name": lecture_name,
                    "id": sentence_id
                }
                data_objects.append(data_object)

            # Insert data objects into the collection
            with client.batch as batch:
                for data_object in data_objects:
                    batch.add_data_object(
                        data_object,
                        class_name=collection_name
                    )
            print(f"Inserted {len(data_objects)} sentences from {file_name} into collection '{collection_name}'.")

# Populate the Weaviate collection
try:
    process_and_insert_sentences_from_json_array(DATA_DIR, "Raw_Text_Sentences")
    print("All sentences have been inserted successfully!")
except Exception as e:
    print(f"Error during processing: {e}")
finally:
    client.close()
