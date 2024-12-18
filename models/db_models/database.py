import weaviate
import weaviate.classes as wvc
import weaviate.classes.config as wc
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

WCD_URL = os.getenv("WCD_URL")
WCD_API_KEY = os.getenv("WCD_API_KEY")

# Weaviate Connection
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WCD_URL,
    auth_credentials=wvc.Auth.api_key(WCD_API_KEY),
)

def create_sentence_collection(client, collection_name="raw_text_file_sentences"):
    """
    Create a Weaviate collection for storing embeddings of sentences.
    """
    try:
        # Define the collection properties
        properties = [
            wc.Property(name="content", data_type="text"),         # Sentence text
            wc.Property(name="file_name", data_type="text"),       # Source file name
        ]

        # Create the collection (no built-in vectorizer, use custom vectors)
        client.collections.create(
            name=collection_name,
            properties=properties,
            vectorizer_config=None  # Disable automatic vectorization
        )
        print(f"Collection '{collection_name}' created successfully!")

    except Exception as e:
        print(f"Error creating collection: {e}")

# Run the collection creation function
create_sentence_collection(client)
client.close()
