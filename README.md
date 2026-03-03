# Advanced ML 02460

## VAE Combined

Run the combined VAE with different priors (gaussian, MoG, flow).

### Output structure

By default, outputs are saved under `outputs/`:

```
outputs/
├── models/          # Model checkpoints (model_gaussian.pt, model_mog.pt, model_flow.pt)
├── samples/         # Generated sample images (samples_gaussian.png, etc.)
└── plots/           # Loss curves (loss_curve_gaussian.png, etc.)
```

Use `--output-dir` to change the base directory.

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

# Custom output directory
python vae_combined.py train --prior gaussian --output-dir my_runs
```

### Sampling

```bash
# Load trained model and generate samples (uses defaults: outputs/models/, outputs/samples/)
python vae_combined.py sample --prior gaussian
python vae_combined.py sample --prior mog
python vae_combined.py sample --prior flow

# Explicit paths
python vae_combined.py sample --prior mog --model outputs/models/model_mog.pt --samples outputs/samples/samples_mog.png
```

### Defaults

- Output dir: `outputs/`
- Model: `outputs/models/model_{prior}.pt`
- Samples: `outputs/samples/samples_{prior}.png`
- Loss plot: `outputs/plots/loss_curve_{prior}.png`

### Quick test

```bash
python vae_combined.py train --prior gaussian --epochs 2 --batch-size 128
```

## DDPM
Run the DDPM to generate mnist samples.

### Training

```bash
python ddpm.py train --batch-size 256 --epochs 20
```

### Sampling

```bash
python ddpm.py sample --batch-size 256 
```


## Latent DDPM
Run the latent DDPM using the trained VAE encoder to generate samples in the latent space.

### Training

```bash
python latent_ddpm.py train --vae-model outputs/models/model_gaussian_original_beta0.999_run1.pt --batch-size 256 --epochs 50
```

### Sampling

```bash
python latent_ddpm.py sample --model outputs/models/model_latent_ddpm.pt --vae-model outputs/models/model_gaussian_original_beta0.999_run1.pt
```


## Sampling wall clock times
Table of wall clock times for sampling 4 images from the trained models, 1000 speed-total and 256 speed-batch:
| Model           | Samples per second |
|-----------------|--------------------|
| VAE (Gaussian)  | 174910.8           |
| VAE (MoG)       | -                  |
| VAE (Flow)      | -                  |
| DDPM            | 4.620              |
| Latent DDPM     | 72.27              | 