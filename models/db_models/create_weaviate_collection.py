import weaviate
import weaviate.classes as wvc
import weaviate.classes.config as wc
from dotenv import load_dotenv
import certifi
import os

load_dotenv()
os.environ['SSL_CERT_FILE'] = certifi.where()

# Define Weaviate Cloud URL and API Key
wcd_url = os.getenv("WCD_URL")
wcd_api_key = os.getenv("WCD_API_KEY")
# openai_key = os.getenv("OPENAI_KEY")
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
    headers=headers,
    additional_config={"verify_ssl": False}
)

try:
    client.collections.create(
        name="Sentences",
        properties=[
            wc.Property(name="sentence", data_type=wc.DataType.TEXT),
            wc.Property(name="lecture_name", data_type=wc.DataType.TEXT),
            wc.Propert(name="id", data_type=wc.DataType.INT)
        ],
        # vectorizer_config=[
        #     wc.Configure.NamedVectors.text2vec_openai(
        #         name="sentence_vector",
        #         source_properties=["sentence"],
        #         model="text-embedding-3-small",
        #         dimensions=3072
        #     ),
        # ],
        vectorizer_config=[
            wc.Configure.NamedVectors.text2vec_huggingface(
                name="sentence_vector",
                source_properties=["sentence"],
                model="sentence-transformers/all-MiniLM-L6-v2",  # Hugging Face model
                dimensions=384  # Adjust based on the chosen model
            ),
        ],
    )
    print("Collection 'Sentences' created successfully with hugging face vectorizer!")
except Exception as e:
    print(f"Error creating collection: {e}")

finally:
    client.close()

