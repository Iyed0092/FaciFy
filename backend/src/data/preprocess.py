import os
from PIL import Image

def resize_images(input_dir, output_dir, size=(128, 128)):
    """
    Resize all images in input_dir and save them to output_dir
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_dir, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(size, Image.BICUBIC)
                img.save(os.path.join(output_dir, filename))
                count += 1
            except Exception as e:
                print(f"Skipping {filename}: {e}")
    print(f"Processed {count} images from {input_dir} -> {output_dir}")

if __name__ == "__main__":
    resize_images("data/raw/faces", "data/processed/faces")
    resize_images("data/raw/anime", "data/processed/anime")
