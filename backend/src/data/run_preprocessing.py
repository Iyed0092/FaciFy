import torch
from pathlib import Path
from PIL import Image
import numpy as np
import sys

STYLEGAN2_PKL = Path(__file__).resolve().parents[1] / "models" / "stylegan2-anime" / "stylegan2-anime.pkl"
OUT_PATH = Path(".") / "anime_sample.png"
sys.path.append(str(STYLEGAN2_PKL.parents[0]))

try:
    import legacy
    import dnnlib
except ImportError:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with dnnlib.util.open_url(str(STYLEGAN2_PKL)) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(DEVICE)

z = torch.randn([1, G.z_dim], device=DEVICE)
c = None if G.c_dim == 0 else torch.zeros([1, G.c_dim], device=DEVICE)

with torch.no_grad():
    img = G(z, c, truncation_psi=0.7, noise_mode='const')

img = (img.clamp(-1, 1) + 1) * 127.5
img = img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
Image.fromarray(img).save(OUT_PATH)
print(f"Saved anime sample at {OUT_PATH.resolve()}")
