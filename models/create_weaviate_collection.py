import weaviate
import weaviate.classes as wvc
import weaviate.classes.config as wc
from dotenv import load_dotenv
import os

load_dotenv()

# Define Weaviate Cloud URL and API Key
wcd_url = os.getenv("WCD_URL")
wcd_api_key = os.getenv("WCD_API_KEY")
openai_key = os.getenv("OPENAI_KEY")
huggingface_key = os.getenv("HF_KEY")

# Set the headers for Hugging Face API
# headers = {
#     "X-HuggingFace-Api-Key": huggingface_key,
# }

headers = {
    "X-OpenAI-Api-Key": openai_key,
}

# Connect to a WCS instance
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,                                 
    auth_credentials=wvc.init.Auth.api_key(wcd_api_key),    
    headers=headers
)

try:
    client.collections.create(
        name="Preprocessed_Chunks",
        properties=[
            wc.Property(name="chunk", data_type=wc.DataType.TEXT),
            wc.Property(name="lecture_name", data_type=wc.DataType.TEXT),
        ],
        # Define & configure the vectorizer module
        vectorizer_config=[
            wc.Configure.NamedVectors.text2vec_openai(
                name="chunk_vector",
                source_properties=["chunk"],
                model="text-embedding-3-small",
                dimensions=3072
            ),
        ],
    )
    print("Collection 'Preprocessed_Chunks' created successfully with openai vectorizer!")
except Exception as e:
    print(f"Error creating collection: {e}")

finally:
    client.close()

