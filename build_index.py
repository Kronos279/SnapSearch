import json
import numpy as np
import hnswlib
import os
from sentence_transformers import SentenceTransformer
folder_path = "/Volumes/Micron/Images"
json_path = os.path.join(folder_path, "captions_embeddings.json")

with open(json_path, "r") as f:
    data = json.load(f)

ids = list(data.keys())
embeddings = np.array([data[i]["embeddings"] for i in ids], dtype="float32")

dim = len(embeddings[0])
num_elements = len(embeddings)

print("Dimension of embeddings: {}".format(dim))
print("Number of elements: {}".format(num_elements))

# Initialize the index
index = hnswlib.Index(space='cosine', dim=dim)

# Create the index
index.init_index(max_elements=num_elements, ef_construction=200, M=16)

# Add embeddings
index.add_items(embeddings, np.arange(num_elements))

# Optional: save for reuse
index.save_index("image_index.bin")


query = "snow"
embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
# Encode the query into the same embedding space
query_embedding = embedder.encode(query, normalize_embeddings=True)  # must output same dimension as your image embeddings
query_embedding = np.array([query_embedding], dtype="float32")

# Load the index (if needed)
index.load_index("image_index.bin")

k = len(ids)  # search across all
labels, distances = index.knn_query(query_embedding, k=k)

# ---- Apply similarity threshold ----
similarity_threshold = 0.26
similar_items = []

for idx, dist in zip(labels[0], distances[0]):
    similarity = 1 - dist
    if similarity >= similarity_threshold:
        similar_items.append({
            "image_id": ids[idx],
            "filename": data[ids[idx]]["filename"],
            "caption": data[ids[idx]]["caption"],
            "similarity": round(similarity, 4)
        })

# ---- Display results ----
if similar_items:
    print(f"\n🧠 Found {len(similar_items)} matching images:")
    for item in similar_items:
        print(f"- {item['filename']} ({item['similarity']}) → {item['caption']}")
else:
    print("No similar images found above threshold.")