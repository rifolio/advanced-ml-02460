# Advanced ML 02460

## VAE Combined

Run the combined VAE with different priors (gaussian, MoG, flow).

### Training

```bash
# Gaussian prior (default)
python vae_combined.py train

# MoG prior
python vae_combined.py train --prior mog

# Flow prior
python vae_combined.py train --prior flow

# With options
python vae_combined.py train --prior mog --epochs 20 --batch-size 64 --latent-dim 64
python vae_combined.py train --prior flow --flow-steps 8 --flow-hidden 256
python vae_combined.py train --prior gaussian --device cuda
```

### Sampling

```bash
# Load trained model and generate samples
python vae_combined.py sample --prior gaussian --model model_gaussian.pt --samples samples_gaussian.png
python vae_combined.py sample --prior mog --model model_mog.pt --samples samples_mog.png
python vae_combined.py sample --prior flow --model model_flow.pt --samples samples_flow.png
```

### Defaults

- Model: `model_{prior}.pt` (e.g. `model_mog.pt`)
- Samples: `samples_{prior}.png`
- Loss plot: `loss_curve_{prior}.png`

### Quick test

```bash
python vae_combined.py train --prior gaussian --epochs 2 --batch-size 128
```
