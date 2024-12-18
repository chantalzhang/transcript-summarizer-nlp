import weaviate as wc
import weaviate.classes as wvc
import time
import os
import requests
import json
from dotenv import load_dotenv
import os
  
load_dotenv()

wcd_url = os.getenv("WCD_URL")
wcd_api_key = os.getenv("WCD_API_KEY")
huggingface_key = os.getenv("HF_KEY")
headers = {
    "X-HuggingFace-Api-Key": huggingface_key,
}

# Connect to a WCS instance
client = wc.connect_to_weaviate_cloud(
    cluster_url=wcd_url,                             
    auth_credentials=wvc.init.Auth.api_key(wcd_api_key),  
    headers=headers
)

# ===== perform near seach =====
try:
    collection = client.collections.get("Raw_Text_Sentences")

    # Perform a near_text query with the semantic similarity score
    response = collection.query.near_text(
        query="I think my and David's office hours have been posted",  # The query will be vectorized by the model provider
        distance=0.8,   # max accepted distance (default distance metric is cosine)
        limit=5
    )

    for obj in response.objects:
        sentence = obj.properties["sentence"]
        print(f"Sentence: {sentence}")

finally:
    client.close()  # Close client gracefully



