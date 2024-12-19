import weaviate
import weaviate.classes as wvc
import weaviate.classes.config as wc
import os
from dotenv import load_dotenv

load_dotenv()

# Define Weaviate Cloud URL and API Key
wcd_url = os.getenv("WCD_URL")
wcd_api_key = os.getenv("WCD_API_KEY")
huggingface_key = os.getenv("HF_KEY")

# Set the headers for Hugging Face API
headers = {
    "X-HuggingFace-Api-Key": huggingface_key,
}

# headers = {
#     "X-OpenAI-Api-Key": openai_key,
# }

# Connect to a WCS instance
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,                                 
    auth_credentials=wvc.init.Auth.api_key(wcd_api_key),    
    headers=headers
)

try:
    client.collections.create(
        name="Sentences",
        properties=[
            wc.Property(name="sentence", data_type=wc.DataType.TEXT),
            wc.Property(name="lecture", data_type=wc.DataType.TEXT),
            wc.Property(name="id", data_type=wc.DataType.TEXT),
        ],
        # Define & configure the vectorizer module
        vectorizer_config=[
            wc.Configure.NamedVectors.text2vec_huggingface(
                name="sentence_vector",
                source_properties=["sentence"],
                model="sentence-transformers/all-MiniLM-L6-v2",  # Hugging Face model
                dimensions=384  # Adjust based on the chosen model
            ),
        ],
        # vectorizer_config=[
        #     # Vectorize the movie title
        #     wc.Configure.NamedVectors.text2vec_openai(
        #         name="sentence", 
        #         source_properties=["question"], 
        #         # choosing openai model
        #         model="text-embedding-3-large",
        #         dimensions=1024
        #     ),
        # ],
        # Define the generative module
        # generative_config=wc.Configure.Generative.openai(),
    )

finally:
    # Close the client connection when done
    client.close()

