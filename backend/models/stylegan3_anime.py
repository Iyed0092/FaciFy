# backend/models/stylegan3_anime.py
import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STYLEGAN3_DIR = REPO_ROOT / "stylegan3"
CHECKPOINT = STYLEGAN3_DIR / "stylegan3-r-afhqv2-512x512.pkl"
OUT = Path(".") / "stylegan3_anime_sample.png"

if str(STYLEGAN3_DIR) not in sys.path:
    sys.path.insert(0, str(STYLEGAN3_DIR))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_generator(pkl_path: Path):
    try:
        import stylegan3.dnnlib as dnnlib
        import stylegan3.legacy as legacy
    except Exception:
        raise RuntimeError("stylegan3 repo (legacy/dnnlib) not found at expected path")
    with dnnlib.util.open_url(str(pkl_path)) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(DEVICE)
    G.eval()
    return G

def tensor_to_pil(img_tensor: torch.Tensor):
    t = img_tensor.detach().cpu()
    t = (t.clamp(-1, 1) + 1) * (255 / 2)
    arr = t.permute(0, 2, 3, 1).numpy().astype(np.uint8)[0]
    return Image.fromarray(arr)

def generate(seed: int = 42, truncation: float = 0.7, out_path: Path = OUT):
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    print(f"[INFO] Loading checkpoint from: {CHECKPOINT}")
    G = load_generator(CHECKPOINT)
    print(f"[INFO] Model loaded. z_dim={G.z_dim}, c_dim={getattr(G,'c_dim',0)}. Device={DEVICE}")
    g = torch.Generator(device=DEVICE)
    g.manual_seed(int(seed))
    z = torch.randn([1, G.z_dim], generator=g, device=DEVICE)
    c = None if getattr(G, "c_dim", 0) == 0 else torch.zeros([1, G.c_dim], device=DEVICE)
    with torch.no_grad():
        out = G(z, c, truncation_psi=float(truncation), noise_mode='const')
        img = out[0] if isinstance(out, tuple) else out
    pil = tensor_to_pil(img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)
    return out_path

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trunc", type=float, default=0.7)
    p.add_argument("--out", type=str, default=str(OUT))
    args = p.parse_args()
    path = generate(seed=args.seed, truncation=args.trunc, out_path=Path(args.out))
    print(f"Saved sample to {path.resolve()}")
