import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np

STYLEGAN3_REPO = Path(__file__).resolve().parents[1] / "models" / "stylegan3"
STYLEGAN3_PKL = STYLEGAN3_REPO / "stylegan3-r-afhqv2-512x512.pkl"
OUT_PATH = Path(".") / "sample.png"

if str(STYLEGAN3_REPO) not in sys.path:
    sys.path.insert(0, str(STYLEGAN3_REPO))

try:
    import legacy
    import dnnlib
except ImportError:
    print(f"[ERROR] Cannot import StyleGAN3 repo from {STYLEGAN3_REPO}")
    raise

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with dnnlib.util.open_url(str(STYLEGAN3_PKL)) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(DEVICE)

z = torch.randn([1, G.z_dim], device=DEVICE)
c = None if G.c_dim == 0 else torch.zeros([1, G.c_dim], device=DEVICE)

with torch.no_grad():
    img = G(z, c, truncation_psi=0.7, noise_mode='const')

img = (img.clamp(-1, 1) + 1) * (255 / 2)
img = img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
Image.fromarray(img).save(OUT_PATH)
print(f"[INFO] Saved sample to {OUT_PATH.resolve()}")
