import argparse
import glob
import numpy as np
import os
import re
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = [SCRIPT_DIR]


def _path(p: str) -> str:
    """Resolve path relative to base dir if not absolute."""
    if os.path.isabs(p):
        return p
    return os.path.join(_BASE_DIR[0], p)


def _models_dir() -> str:
    """Path to outputs/models."""
    return _path("outputs/models")


from fid import compute_fid
from ddpm import DDPM
from latent_ddpm import DDPM as LatentDDPM
from unet import Unet, LatentUnet
from vae_combined import VAE, GaussianPrior, GaussianEncoder, GaussianDecoder


def _beta_to_label(beta_str: str) -> str:
    """Convert filename beta string to display label."""
    labels = {"0.999": r"$\beta=1$", "1e-06": r"$\beta=10^{-6}$", "0.001": r"$\beta=10^{-3}$"}
    return labels.get(beta_str, rf"$\beta={beta_str}$")


def discover_vae_models() -> list[tuple[str, str]]:
    """Discover Gaussian original VAE models. Returns [(beta_str, path), ...] sorted by beta_str."""
    pattern = os.path.join(_models_dir(), "model_gaussian_original_beta*.pt")
    found = []
    for p in glob.glob(pattern):
        m = re.search(r"model_gaussian_original_beta([\d.e+-]+)(?:_run\d+)?\.pt$", os.path.basename(p))
        if m:
            found.append((m.group(1), p))
    return sorted(found, key=lambda x: float(x[0]))


def discover_latent_ddpm() -> list[tuple[str, str, str, str]]:
    """Discover L-DDPM triples (ddpm, vae, stats). Returns [(beta_str, ddpm_path, vae_path, stats_path), ...]."""
    vae_by_beta = {beta: path for beta, path in discover_vae_models()}
    pattern = os.path.join(_models_dir(), "*_model_latent_ddpm.pt")
    found = []
    for ddpm_path in glob.glob(pattern):
        base = os.path.basename(ddpm_path)
        m = re.match(r"^([\d.e+-]+)_model_latent_ddpm\.pt$", base)
        if not m:
            continue
        beta_str = m.group(1)
        stats_path = os.path.join(_models_dir(), f"{beta_str}_latent_stats.pt")
        vae_path = vae_by_beta.get(beta_str)
        if os.path.exists(stats_path) and vae_path:
            found.append((beta_str, ddpm_path, vae_path, stats_path))
    return sorted(found, key=lambda x: float(x[0]))


def get_mnist_test_loader(batch_size=64):
    """Load MNIST test set with values in [-1, 1] and shape (N, 1, 28, 28)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - 0.5) * 2.0),
    ])
    dataset = datasets.MNIST(_path("data/"), train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_ddpm_samples(n_samples, model_path, device, batch_size=64):
    """Generate samples from DDPM. Output is already in [-1, 1]. Returns (samples, elapsed_sec)."""
    D = 28 * 28
    network = Unet()
    model = DDPM(network, T=1000).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    samples = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in tqdm(range(0, n_samples, batch_size), desc="DDPM sampling"):
            b = min(batch_size, n_samples - len(samples))
            s = model.sample((b, D)).cpu()
            s = s.view(-1, 1, 28, 28)
            samples.append(s)
    elapsed = time.perf_counter() - t0
    return torch.cat(samples, dim=0)[:n_samples], elapsed


def load_vae(model_path, latent_dim, device):
    """Build and load Gaussian VAE (original MNIST, Gaussian decoder)."""
    prior = GaussianPrior(latent_dim)
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, latent_dim * 2),
    )
    decoder_net = nn.Sequential(
        nn.Linear(latent_dim, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 784), nn.Unflatten(-1, (28, 28)),
    )
    encoder = GaussianEncoder(encoder_net)
    decoder = GaussianDecoder(decoder_net)
    vae = VAE(prior, decoder, encoder, beta=1.0).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()
    return vae


def load_latent_ddpm_samples(n_samples, ddpm_path, vae_path, latent_stats_path,
                             latent_dim, device, batch_size=64):
    """Generate samples from Latent DDPM via VAE decoder. VAE outputs [0,1] -> convert to [-1,1].
    Returns (samples, elapsed_sec)."""
    network = LatentUnet()
    model = LatentDDPM(network, T=1000).to(device)
    model.load_state_dict(torch.load(ddpm_path, map_location=device))
    model.eval()

    vae = load_vae(vae_path, latent_dim, device)
    stats = torch.load(latent_stats_path)

    samples = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in tqdm(range(0, n_samples, batch_size), desc="L-DDPM sampling"):
            b = min(batch_size, n_samples - len(samples))
            z = model.sample((b, 1, 8, 8)).cpu()
            z = z * stats["z_std"] + stats["z_mean"]
            z = z.view(b, -1).to(device)
            x = vae.decoder(z).sample().cpu()  # [0, 1]
            x = (x - 0.5) * 2.0  # -> [-1, 1]
            x = x.view(-1, 1, 28, 28)
            samples.append(x)
    elapsed = time.perf_counter() - t0
    return torch.cat(samples, dim=0)[:n_samples], elapsed


def load_vae_samples(n_samples, model_path, latent_dim, device, batch_size=64):
    """Generate samples from Gaussian VAE. Output is [0,1] -> convert to [-1,1]. Returns (samples, elapsed_sec)."""
    vae = load_vae(model_path, latent_dim, device)

    samples = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in tqdm(range(0, n_samples, batch_size), desc="VAE sampling"):
            b = min(batch_size, n_samples - len(samples))
            x = vae.sample(b).cpu()
            x = (x - 0.5) * 2.0  # [0,1] -> [-1,1]
            x = x.view(-1, 1, 28, 28)
            samples.append(x)
    elapsed = time.perf_counter() - t0
    return torch.cat(samples, dim=0)[:n_samples], elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10000,
                        help="Number of real and generated samples for FID (default: 10000)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--classifier", type=str, default="mnist_classifier.pth")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Base directory for paths (e.g. /content in Colab). Default: script dir.")
    parser.add_argument("--ddpm", type=str, default="outputs/models/model_ddpm.pt")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--models", type=str, default="ddpm,latent_ddpm,vae",
                        help="Comma-separated: ddpm,latent_ddpm,vae")
    args = parser.parse_args()

    if args.base_dir is not None:
        _BASE_DIR[0] = os.path.abspath(args.base_dir)
        print(f"Using base dir: {_BASE_DIR[0]}")

    device = args.device
    n = args.n_samples

    # Results: list of (model_name, fid, samples_per_sec)
    results = []

    # Load real test images
    print("Loading MNIST test set...")
    loader = get_mnist_test_loader(batch_size=args.batch_size)
    real_images = []
    for x, _ in loader:
        real_images.append(x)
        if sum(b.shape[0] for b in real_images) >= n:
            break
    x_real = torch.cat(real_images, dim=0)[:n]

    ddpm_path = _path(args.ddpm)
    if "ddpm" in args.models and os.path.exists(ddpm_path):
        print("\n--- DDPM ---")
        x_gen, elapsed = load_ddpm_samples(n, ddpm_path, device, args.batch_size)
        fid = float(np.real(compute_fid(x_real, x_gen, device=device, classifier_ckpt=_path(args.classifier))))
        sps = n / elapsed
        results.append(("DDPM", fid, sps))
        print(f"DDPM FID: {fid:.4f}  |  Samples/s: {sps:.2f}")

    if "latent_ddpm" in args.models:
        ldpm_configs = discover_latent_ddpm()
        for beta_str, ddpm_path, vae_path, stats_path in ldpm_configs:
            label = _beta_to_label(beta_str)
            print(f"\n--- L-DDPM({label}) ---")
            x_gen, elapsed = load_latent_ddpm_samples(
                n, ddpm_path, vae_path, stats_path,
                args.latent_dim, device, args.batch_size,
            )
            fid = float(np.real(compute_fid(x_real, x_gen, device=device, classifier_ckpt=_path(args.classifier))))
            sps = n / elapsed
            results.append((f"L-DDPM({label})", fid, sps))
            print(f"L-DDPM({label}) FID: {fid:.4f}  |  Samples/s: {sps:.2f}")

    if "vae" in args.models:
        vae_configs = discover_vae_models()
        for beta_str, vae_path in vae_configs:
            label = _beta_to_label(beta_str)
            print(f"\n--- Gaus. VAE({label}) ---")
            x_gen, elapsed = load_vae_samples(n, vae_path, args.latent_dim, device, args.batch_size)
            fid = float(np.real(compute_fid(x_real, x_gen, device=device, classifier_ckpt=_path(args.classifier))))
            sps = n / elapsed
            results.append((f"Gaus. VAE({label})", fid, sps))
            print(f"Gaus. VAE({label}) FID: {fid:.4f}  |  Samples/s: {sps:.1f}")

    print("\n" + "=" * 50)
    print("FID & Sampling Speed Summary (lower FID is better)")
    print("=" * 50)
    for name, fid, sps in results:
        print(f"  {name}:  FID={fid:.4f}  |  Samples/s={sps:.2f}")


if __name__ == "__main__":
    main()
