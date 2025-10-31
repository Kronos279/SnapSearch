import torch
from PIL import Image
import json
import os
import time
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForCausalLM
from preprocessing_images import renameAllImages
from build_index import updateIndex

renameAllImages()

avg_time =[]

start_all_time = time.perf_counter()

# Use MPS if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load Florence-2 processor and model
model_id = "microsoft/Florence-2-base"

# Use float16 for better performance on MPS
dtype = torch.float16 if device == "mps" else torch.float32

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=dtype,
    attn_implementation="eager"
).to(device)

embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

# Load your local image
folder_path = "/Volumes/Micron/Images"
json_path = os.path.join(folder_path, "captions_embeddings.json")
extensions = (".jpg", ".jpeg", ".png", ".heic", ".png", ".webp")

data = {}
if os.path.exists(json_path):
    try:
        with open(json_path, "r") as f:
            content = f.read().strip()
            if content:
                data = json.loads(content)
            else:
                print("JSON file is empty — starting fresh.")
    except json.JSONDecodeError:
        print("Corrupted JSON detected — resetting file.")

for filename in os.listdir(folder_path):

    if not filename.lower().endswith(extensions):
        continue

    image_id = filename.split(".")[0]
    if image_id in data:
        print(f"Already processed Image {image_id}") # Uncomment if you want this log
        continue

    image_path = os.path.join(folder_path, filename)

    raw_image = None

    try:
        raw_image = Image.open(image_path).convert("RGB")
        max_size = 128 # You can experiment: 384, 512, or 640
        raw_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    except:
        print(f"Failed to process Image {image_id}")

    if (raw_image is None):
        continue

    # Use the detailed caption prompt
    prompt = "<DETAILED_CAPTION>"

    # Process the image and prompt
    inputs = processor(text=prompt, images=raw_image, return_tensors="pt")

    # Fix 2: Correctly move tensors to device
    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
        for k, v in inputs.items()
    }

    # --- Timer Start ---
    start_time = time.perf_counter()

    # Generate
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=1,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            use_cache=False
        )

    # Decode and clean the output
    generated_text = processor.batch_decode(out, skip_special_tokens=True)[0]
    embedding =None
    try:
        caption = generated_text.split(">", 1)[-1].strip()
        caption_text = caption.strip()
        embedding = embedder.encode(caption_text, normalize_embeddings=True).tolist()
    except:
        print(f"Failed to parse caption for {image_id}")
        caption = ""

    data[image_id] = {
        "filename": filename,
        "caption": caption,
        "embeddings": embedding,

    }

    # Save after each image
    with open(json_path, "w") as f:
        json.dump(data, f, separators=(',', ':'), ensure_ascii=False)

    # --- Timer End ---
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Generation time: {total_time}")

end_all_time = time.perf_counter()


total_time_for_all = end_all_time - start_all_time
print("Total Time it took for everything",total_time_for_all)


updateIndex()
