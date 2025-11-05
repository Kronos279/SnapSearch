import json
import numpy as np
import hnswlib
import os
from sentence_transformers import SentenceTransformer

# --- Global Constants & Model Initialization ---
# The dimension of the embeddings (all-MiniLM-L6-v2 is 384)
EMBEDDER_DIM = 384

# Load the sentence transformer model once when the script is imported
# This is a global, read-only object.
print("Loading SentenceTransformer model (for search)...")
try:
    EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
    print("Search model loaded successfully.")
except Exception as e:
    print(f"Fatal error: Could not load model. {e}")
    EMBEDDER = None

# --- In-Memory Cache ---
# These globals will hold the index/data for the *currently active* folder
# to make searching instant.
DATA_CACHE = {}
IDS_CACHE = []
INDEX_CACHE = None
CURRENT_FOLDER_LOADED = None


# --- Manifest Helper Functions ---

def load_manifest(folder_path: str) -> list[str]:
    """Loads the list of indexed image IDs from a folder's manifest file."""
    manifest_path = os.path.join(folder_path, "index_manifest.json")
    if not os.path.exists(manifest_path):
        return []  # No manifest, so no items are indexed
    try:
        with open(manifest_path, "r") as f:
            data = json.load(f)
            return data.get("indexed_ids", [])
    except Exception as e:
        print(f"Warning: Could not read manifest {manifest_path}. Error: {e}")
        return []


def save_manifest(folder_path: str, ids_list: list[str]):
    """Saves the list of indexed image IDs to a folder's manifest file."""
    manifest_path = os.path.join(folder_path, "index_manifest.json")
    try:
        with open(manifest_path, "w") as f:
            json.dump({"indexed_ids": ids_list}, f)
    except Exception as e:
        print(f"Error saving manifest {manifest_path}: {e}")


# --- Main Indexing Function ---

def updateIndex(folder_path: str):
    """
    Checks a folder's JSON for new embeddings and incrementally updates
    the HNSW index file (image_index.bin) for that folder.
    """
    global CURRENT_FOLDER_LOADED
    print(f"Checking for index updates in: {folder_path}")

    # Define all paths relative to the specific folder
    json_path = os.path.join(folder_path, "captions_embeddings.json")
    index_path = os.path.join(folder_path, "image_index.bin")

    if not os.path.exists(json_path):
        print(f"Error: captions_embeddings.json not found in {folder_path}.")
        return

    # 1. Load the data JSON
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data for update: {e}")
        return

    # 2. Find out what's new
    all_json_ids = set(data.keys())
    indexed_ids = load_manifest(folder_path)  # IDs already in image_index.bin
    indexed_ids_set = set(indexed_ids)
    new_ids_to_add = list(all_json_ids - indexed_ids_set)

    if not new_ids_to_add:
        print(f"Index is already up-to-date for: {folder_path}")
        return

    print(f"Found {len(new_ids_to_add)} new items to add to the index.")

    # 3. Get embeddings for *only* the new items
    new_embeddings = []
    valid_new_ids = []  # IDs we are actually adding
    for img_id in new_ids_to_add:
        emb = data[img_id].get("embeddings")
        if emb and len(emb) == EMBEDDER_DIM:
            new_embeddings.append(emb)
            valid_new_ids.append(img_id)
        else:
            print(f"Warning: Skipping item {img_id} (missing or invalid embedding)")

    if not valid_new_ids:
        print("No valid new items to add.")
        return

    new_embeddings_np = np.array(new_embeddings, dtype="float32")

    # 4. Load existing index or create a new one
    current_num_elements = len(indexed_ids)
    new_total_elements = current_num_elements + len(valid_new_ids)

    # Initialize the index object
    index = hnswlib.Index(space='cosine', dim=EMBEDDER_DIM)

    if os.path.exists(index_path) and current_num_elements > 0:
        # Load the existing index
        print(f"Loading existing index with {current_num_elements} items...")
        index.load_index(index_path, max_elements=current_num_elements)
        # Make space for the new items
        index.resize_index(new_total_elements)
    else:
        # Create a new index
        print(f"Creating new index for {new_total_elements} elements...")
        index.init_index(max_elements=new_total_elements, ef_construction=200, M=16)

    # 5. Add *only* the new items
    # The new items will have numerical labels from `current_num_elements` up to `new_total_elements - 1`
    new_numerical_labels = np.arange(current_num_elements, new_total_elements)

    print(f"Adding {len(valid_new_ids)} new items to index...")
    index.add_items(new_embeddings_np, new_numerical_labels)

    # 6. Save the updated index and manifest
    index.save_index(index_path)

    indexed_ids.extend(valid_new_ids)  # Add the new IDs to the master list
    save_manifest(folder_path, indexed_ids)

    print(f"Indexing complete. Index saved with {new_total_elements} total items.")

    # Clear the in-memory cache if it was for this folder
    if CURRENT_FOLDER_LOADED == folder_path:
        CURRENT_FOLDER_LOADED = None
        global DATA_CACHE, IDS_CACHE, INDEX_CACHE
        DATA_CACHE = {}
        IDS_CACHE = []
        INDEX_CACHE = None


# --- Search Functions ---

def load_folder_into_memory(folder_path: str) -> bool:
    """
    Loads a specific folder's index and data files into the global cache
    for fast, repeated searching. Your GUI should call this when a
    user clicks on a folder.
    """
    global DATA_CACHE, IDS_CACHE, INDEX_CACHE, CURRENT_FOLDER_LOADED

    if CURRENT_FOLDER_LOADED == folder_path:
        print(f"'{folder_path}' is already in the cache.")
        return True  # Already loaded

    print(f"Loading '{folder_path}' into memory cache...")

    # Define paths
    json_path = os.path.join(folder_path, "captions_embeddings.json")
    index_path = os.path.join(folder_path, "image_index.bin")

    try:
        # 1. Load the ID manifest
        IDS_CACHE = load_manifest(folder_path)
        if not IDS_CACHE:
            print(f"Folder '{folder_path}' has no indexed items.")
            CURRENT_FOLDER_LOADED = None  # Mark as nothing loaded
            return False

        # 2. Load the HNSW index
        INDEX_CACHE = hnswlib.Index(space='cosine', dim=EMBEDDER_DIM)
        INDEX_CACHE.load_index(index_path, max_elements=len(IDS_CACHE))

        # 3. Load the data JSON
        with open(json_path, "r") as f:
            DATA_CACHE = json.load(f)

        # 4. Mark the cache as successfully loaded
        CURRENT_FOLDER_LOADED = folder_path
        print(f"Successfully loaded '{folder_path}' into cache ({len(IDS_CACHE)} items).")
        return True

    except Exception as e:
        print(f"Error loading folder '{folder_path}' into cache: {e}")
        # Reset cache on failure
        DATA_CACHE = {}
        IDS_CACHE = []
        INDEX_CACHE = None
        CURRENT_FOLDER_LOADED = None
        return False


def searchSimilaritems(prompt: str) -> list[str]:
    """
    Searches the *currently cached* folder for a text prompt.

    Returns a list of matching filenames.
    """
    global DATA_CACHE, IDS_CACHE, INDEX_CACHE, EMBEDDER, CURRENT_FOLDER_LOADED

    if EMBEDDER is None:
        print("Error: Search model is not loaded. Cannot perform search.")
        return []

    if INDEX_CACHE is None or not CURRENT_FOLDER_LOADED:
        print("Error: No folder is loaded into memory.")
        print("Please call 'load_folder_into_memory(folder_path)' first.")
        return []

    print(f"Searching in '{CURRENT_FOLDER_LOADED}' for: '{prompt}'")
    query_embedding = EMBEDDER.encode(prompt, normalize_embeddings=True)
    query_embedding = np.array([query_embedding], dtype="float32")

    # Search for all neighbors
    k = len(IDS_CACHE)
    try:
        labels, distances = INDEX_CACHE.knn_query(query_embedding, k=k)
    except Exception as e:
        print(f"Error during knn_query: {e}")
        return []

    # --- Filter results ---
    similarity_threshold = 0.35
    similar_filenames = []

    for idx, dist in zip(labels[0], distances[0]):
        similarity = 1 - dist
        if similarity >= similarity_threshold:
            # The numerical label 'idx' is the position in our IDS_CACHE list
            image_id = IDS_CACHE[idx]
            # Get the filename from the data cache
            filename = DATA_CACHE.get(image_id, {}).get("filename")
            if filename:
                similar_filenames.append(filename)
            else:
                print(f"Warning: Indexed ID '{image_id}' not found in data JSON.")

    return similar_filenames


if __name__ == "__main__":
    # You must provide a path to test
    TEST_FOLDER = "/Volumes/Micron/Images"

    # 1. First, update the index.
    # This will find and add any new images from the JSON.
    updateIndex(TEST_FOLDER)

    print("\n" + "=" * 30 + "\n")

    # 2. Next, load that folder into memory for searching.
    if load_folder_into_memory(TEST_FOLDER):

        # 3. Now you can run searches on that folder.
        query = "a dog playing in a park"
        print(f"Searching for: '{query}'")

        results = searchSimilaritems(query)

        if results:
            print(f"Found {len(results)} matching images:")
            for filename in results:
                print(f"- {filename}")
        else:
            print("No similar images found.")

    else:
        print(f"Could not load folder {TEST_FOLDER} to run search.")