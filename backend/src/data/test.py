import numpy as np
data = np.load("data/embeddings/faces.npy", allow_pickle=True).item()
print("Embeddings shape:", data["embeddings"].shape)
print("Number of filenames:", len(data["filenames"]))
