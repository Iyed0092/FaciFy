import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

def extract_facenet_embeddings(
    input_dir: str,
    output_file: str = "data/embeddings/faces.npy",
    device: str = "cuda"
):
    """
    Extract face embeddings using Facenet (InceptionResnetV1) for all images in input_dir
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print("Loading Facenet model...")
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    mtcnn = MTCNN(image_size=160, margin=0, device=device)

    embeddings = []
    filenames = []

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    total_files = len(image_files)
    print(f"Found {total_files} images to process.")

    for idx, filename in enumerate(image_files, 1):
        img_path = os.path.join(input_dir, filename)
        try:
            img = Image.open(img_path).convert("RGB")
            face = mtcnn(img)
            if face is None:
                print(f"Warning: no face detected in {filename}")
                continue

            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(face)

            embeddings.append(embedding.cpu().numpy())
            filenames.append(filename)

            # Debug progress every 100 images
            if idx % 100 == 0 or idx == total_files:
                print(f"Processed {idx}/{total_files} images")

        except Exception as e:
            print(f"Warning: skipped {filename} due to error: {e}")

    embeddings_array = np.vstack(embeddings)
    np.save(output_file, {"embeddings": embeddings_array, "filenames": filenames})

    print(f"\nSaved embeddings for {len(filenames)} images to {output_file}")
    print("Extraction complete!")

# ---------------- Standalone execution ----------------
if __name__ == "__main__":
    extract_facenet_embeddings(
        input_dir="data/processed/faces",
        output_file="data/embeddings/faces.npy",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
