from transformers import AutoTokenizer, AutoModel
import torch

# Sentence-BERT model
model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def generate_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Example: Split lecture into chunks and embed
lecture_chunks = [
    "Artificial Intelligence is transforming industries. Machine learning models analyze data.",
    "NLP enables machines to understand human language. It powers chatbots, translation systems.",
    "Deep learning uses neural networks to solve complex problems in images and videos."
]

# Generate embeddings for each chunk
embeddings = [generate_embedding(chunk) for chunk in lecture_chunks]

# Print results
for i, emb in enumerate(embeddings):
    print(f"Chunk {i+1} Embedding Size: {len(emb)}")

print(embeddings)