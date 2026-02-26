# VAE Experiment Results Summary

## Experiment Configuration
- **Dataset**: MNIST
- **Latent dim**: 64
- **Flow hidden**: 128
- **Flow steps**: 6
- **Epochs**: 20
- **Batch size**: 32
- **Runs per prior**: 3

## Results: Test Set Log-Likelihood (ELBO)

| Prior | Mean | Std | Std Error (SE) |
|-------|------|-----|----------------|
| Gaussian | -91.64 | 0.22 | 0.13 |
| MoG | -89.08 | 0.34 | 0.20 |
| Flow | -89.78 | 0.47 | 0.27 |

## Notes
- Higher (less negative) ELBO indicates better model fit
- MoG prior achieves the best performance
- Std Error = Std / √n (n=3 runs)
