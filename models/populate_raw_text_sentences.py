import os
import weaviate
import weaviate.classes as wvc
from weaviate.util import generate_uuid5 
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
import nltk

# nltk tokenizer data
nltk.download('punkt')

load_dotenv()

DATA_DIR = "../dataset/mycourses/"
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



def process_and_insert_sentences(data_dir, collection_name):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_dir, file_name)
            lecture_name = file_name.replace(".txt", "")
            print(f"Processing {file_path}...")

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            sentences = sent_tokenize(text)

            # Prepare data objects for batch insertion
            data_objects = []
            for sentence in sentences:
                data_object = {
                    "sentence": sentence,
                    "lecture_name": lecture_name,
                }
                data_objects.append(data_object)

            raw_text_sentences = client.collections.get(collection_name)
            with raw_text_sentences.batch.dynamic() as batch:
                for data_row in data_objects:
                    obj_uuid = generate_uuid5(data_row)
                    batch.add_object(
                        properties=data_row,
                        uuid=obj_uuid
            )
            # raw_text_sentences.data.insert_many(data_objects, batch)

# Populate the Weaviate collection
try:
    process_and_insert_sentences(DATA_DIR, "Raw_Text_Sentences")
    print("All sentences have been inserted successfully!")
except Exception as e:
    print(f"Error during processing: {e}")
finally:
    client.close()
