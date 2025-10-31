import json
import numpy as np
import hnswlib
import os
from sentence_transformers import SentenceTransformer

# --- Global Constants & Model Initialization ---
FOLDER_PATH = "/Volumes/Micron/Images"
JSON_PATH = os.path.join(FOLDER_PATH, "captions_embeddings.json")
INDEX_PATH = os.path.join(FOLDER_PATH, "image_index.bin")
# This new file tracks what's IN the index
MANIFEST_PATH = os.path.join(FOLDER_PATH, "index_manifest.json")
EMBEDDER_DIM = 384

print("Loading SentenceTransformer model...")
try:
    EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Fatal error: Could not load model. {e}")
    EMBEDDER = None

# --- Global State ---
DATA = {}
# This IDS list is now our "source of truth" for the index order
IDS = []
INDEX = None


# --- Helper function to load the manifest ---
def load_manifest():
    """Loads the list of IDs currently in the index."""
    if not os.path.exists(MANIFEST_PATH):
        return []
    try:
        with open(MANIFEST_PATH, "r") as f:
            manifest = json.load(f)
            return manifest.get("indexed_ids", [])
    except Exception as e:
        print(f"Warning: Could not read manifest file. Rebuilding index. Error: {e}")
        return []


# --- Helper function to save the manifest ---
def save_manifest(ids_list):
    """Saves the list of IDs currently in the index."""
    try:
        with open(MANIFEST_PATH, "w") as f:
            json.dump({"indexed_ids": ids_list}, f)
    except Exception as e:
        print(f"Error saving manifest file: {e}")


# --- Function 1: Build/Update the Index (FIXED) ---
def updateIndex():
    """
    Loads embeddings from JSON and *incrementally* updates the HNSW index.
    """
    global DATA, IDS, INDEX, EMBEDDER_DIM, JSON_PATH, INDEX_PATH

    print("Checking for new items to index...")
    try:
        with open(JSON_PATH, "r") as f:
            DATA = json.load(f)
    except Exception as e:
        print(f"Error: Could not load data from {JSON_PATH}. {e}")
        return

    # 1. Find out what's new
    all_json_ids = set(DATA.keys())
    IDS = load_manifest()  # Load what's *already* indexed
    indexed_ids_set = set(IDS)

    new_ids_to_add = list(all_json_ids - indexed_ids_set)

    if not new_ids_to_add:
        print("Index is already up-to-date. Nothing to add.")
        # Ensure index is loaded for searching
        if INDEX is None and os.path.exists(INDEX_PATH):
            load_index_from_disk()
        return

    print(f"Found {len(new_ids_to_add)} new items to add to the index.")

    # 2. Get embeddings for *only* the new items
    new_embeddings = []
    valid_new_ids = []
    for i in new_ids_to_add:
        emb = DATA[i].get("embeddings")
        if emb and len(emb) == EMBEDDER_DIM:
            new_embeddings.append(emb)
            valid_new_ids.append(i)  # Keep track of IDs we're *actually* adding
        else:
            print(f"Warning: Skipping new item {i} (missing or invalid embedding)")

    if not valid_new_ids:
        print("No valid new items to add.")
        return

    new_embeddings_np = np.array(new_embeddings, dtype="float32")

    # 3. Load existing index or create a new one
    current_num_elements = len(IDS)
    new_total_elements = current_num_elements + len(valid_new_ids)

    if os.path.exists(INDEX_PATH) and current_num_elements > 0:
        # Load the existing index to add to it
        print(f"Loading existing index with {current_num_elements} items...")
        if INDEX is None:
            load_index_from_disk()

        # Resize the index to make space for new items
        INDEX.resize_index(new_total_elements)
    else:
        # Create a new index
        print(f"Creating new index for {new_total_elements} elements...")
        INDEX = hnswlib.Index(space='cosine', dim=EMBEDDER_DIM)
        INDEX.init_index(max_elements=new_total_elements, ef_construction=200, M=16)

    # 4. Add *only* the new items
    # The new items will have numerical labels starting *after* the old ones
    new_numerical_labels = np.arange(current_num_elements, new_total_elements)

    print(f"Adding {len(valid_new_ids)} new items to index...")
    INDEX.add_items(new_embeddings_np, new_numerical_labels)

    # 5. Save everything
    INDEX.save_index(INDEX_PATH)

    # Update our master list of IDs and save it
    IDS.extend(valid_new_ids)
    save_manifest(IDS)

    print(f"Indexing complete. Index saved with {new_total_elements} total items.")


# --- Helper function to load index for searching ---
def load_index_from_disk():
    global INDEX, IDS, EMBEDDER_DIM, MANIFEST_PATH, INDEX_PATH
    try:
        if not os.path.exists(MANIFEST_PATH):
            print("Error: Manifest file not found. Cannot load index.")
            return False

        IDS = load_manifest()
        if not IDS:
            print("Error: No items in manifest. Cannot load index.")
            return False

        print(f"Loading index with {len(IDS)} items...")
        INDEX = hnswlib.Index(space='cosine', dim=EMBEDDER_DIM)
        INDEX.load_index(INDEX_PATH, max_elements=len(IDS))
        print("Index loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading index: {e}")
        INDEX = None
        IDS = []
        return False


# --- Function 2: Search the Index (Updated) ---
def searchSimilaritems(prompt: str) -> list[str]:
    """
    Searches the index for a given text prompt and returns
    a list of matching filenames.
    """
    global DATA, IDS, INDEX, EMBEDDER

    if EMBEDDER is None:
        print("Error: Model is not loaded.")
        return []

    # --- Load index if not in memory ---
    if INDEX is None:
        if not load_index_from_disk():
            print("Please run 'updateIndex()' first to create the index file.")
            return []

    # --- Load data if not in memory ---
    if not DATA:
        try:
            with open(JSON_PATH, "r") as f:
                DATA = json.load(f)
        except Exception as e:
            print(f"Error loading data file: {e}")
            return []

    # --- Perform the Search ---
    query_embedding = EMBEDDER.encode(prompt, normalize_embeddings=True)
    query_embedding = np.array([query_embedding], dtype="float32")

    k = len(IDS)  # Search all items
    labels, distances = INDEX.knn_query(query_embedding, k=k)

    # --- Filter results ---
    similarity_threshold = 0.26
    similar_filenames = []

    for idx, dist in zip(labels[0], distances[0]):
        similarity = 1 - dist
        if similarity >= similarity_threshold:
            # The numerical label 'idx' directly maps to our 'IDS' list
            image_id = IDS[idx]
            filename = DATA.get(image_id, {}).get("filename")
            if filename:
                similar_filenames.append(filename)
            else:
                print(f"Warning: Indexed ID '{image_id}' not found in data JSON.")

    return similar_filenames


# --- Example of how to use these functions ---
if __name__ == "__main__":
    # 1. Run the update function.
    # This will check for new images and add them.
    # If no new images, it's very fast.
    updateIndex()

    print("\n" + "=" * 30 + "\n")

    # 2. Now you can run searches.
    query = "a dog playing in a park"
    print(f"Searching for: '{query}'")

    results = searchSimilaritems(query)

    if results:
        print(f"Found {len(results)} matching images:")
        for filename in results:
            print(f"- {filename}")
    else:
        print("No similar images found.")