# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from unet import LatentUnet
from vae_combined import VAE, GaussianPrior, GaussianEncoder, BernoulliDecoder, GaussianDecoder
import time
class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        ### Implement Algorithm 1 here ###
        epsilon = torch.randn_like(x)
        t = torch.randint(0, self.T, (x.shape[0],), device=x.device)
        t_norm = t.float() / (self.T - 1)
        t_norm = t_norm.unsqueeze(1)
        alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1, 1) # Reshape to (batch_size, 1, 1, 1) for broadcasting

        network_input = torch.sqrt(alpha_cumprod_t) * x + torch.sqrt(1 - alpha_cumprod_t) * epsilon
        neg_elbo = (epsilon - self.network(network_input, t_norm)).square().sum(dim=(1, 2, 3))

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T-1, -1, -1): # Inverse loop from T-1 to 0
            ### Implement the remaining of Algorithm 2 here ###
            if t > 0:
                t_norm = torch.full((shape[0], 1), t / (self.T - 1), device=self.alpha.device)
                epsilon_theta = self.network(x_t, t_norm)
                mean = (x_t - (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_cumprod[t]) * epsilon_theta) / torch.sqrt(self.alpha[t])
                variance = self.beta[t]
                x_t = mean + torch.sqrt(variance) * torch.randn_like(x_t)
            else:
                t_norm = torch.full((shape[0], 1), t / (self.T - 1), device=self.alpha.device)
                epsilon_theta = self.network(x_t, t_norm)
                mean = (x_t - (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_cumprod[t]) * epsilon_theta) / torch.sqrt(self.alpha[t])
                x_t = mean # When t=0, we do not add noise

        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(epochs), desc="Training")

    losses = []
    epoch_loss = 0
    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()


        epoch_loss /= len(data_loader)
        # Update progress bar
        progress_bar.set_postfix(loss=f"⠀{epoch_loss:12.4f}", epoch=f"{epoch+1}/{epochs}")
        progress_bar.update()
        losses.append(epoch_loss)
        epoch_loss = 0
        
    return losses

if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'debug', 'speed'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--speed-total', type=int, default=1000)
    parser.add_argument('--speed-batch', type=int, default=64)
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--vae-model', type=str, default='model_gaussian.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--latent-stats', type=str, default='latent_stats.pt', help='file to save latent normalization stats to or load normalization stats from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian'], help='prior type (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=64, help='latent dimension (default: %(default)s)')
    parser.add_argument('--mnist-type', type=str, default='original', choices=['binary', 'original'], help='type of MNIST data (default: %(default)s)')
    parser.add_argument('--beta', type=float, default='1', help='beta value for VAE (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    if args.mode == 'train':
        # Loading VAE model
        M = args.latent_dim
        D = 28 * 28
        if args.prior == 'gaussian':
            prior = GaussianPrior(M)

        # Define encoder and decoder networks
        encoder_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, M*2),
        )

        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Unflatten(-1, (28, 28))
        )

        # Define VAE model
        encoder = GaussianEncoder(encoder_net)
        if args.mnist_type == 'binarized':
            decoder = BernoulliDecoder(decoder_net)
        else:            
            decoder = GaussianDecoder(decoder_net)

        vae_model = VAE(prior, decoder, encoder, beta=args.beta).to(args.device)
        vae_model.load_state_dict(torch.load(args.vae_model, map_location=torch.device(args.device)))

        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.squeeze())])

        mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True, transform=transform), batch_size=args.batch_size, shuffle=True)
        # Pre-compute the latent representations of the training data using the VAE encoder
        train_data = []
        for x, _ in tqdm(mnist_train_loader, desc="Encoding training data with VAE"):
            x = x.to(args.device)
            z = vae_model.encoder(x)
            z = z.rsample() # Sample from the encoder distribution
            z = nn.Unflatten(-1, (8,8))(z)
            train_data.append(z.detach().cpu())
        train_data = torch.cat(train_data, dim=0).unsqueeze(1) # Shape: (num_samples, 1, latent_dim)
        
        # Normalize latent representations to be in the range [0, 1]
        z_mean = train_data.mean()
        z_std = train_data.std()
        train_data = (train_data - z_mean) / z_std

        # Save for sampling
        torch.save({'z_mean': z_mean, 'z_std': z_std}, args.latent_stats)
        
        print(f"Encoded training data shape: {train_data.shape}") # Should be (num_samples, 1, latent_dim)
        latent_dataset = torch.utils.data.TensorDataset(train_data)
        latent_loader = torch.utils.data.DataLoader(latent_dataset, batch_size=args.batch_size, shuffle=True)

        # DDPM network
        network = LatentUnet()

        # Set the number of steps in the diffusion process
        T = 1000

        # Define model
        model = DDPM(network, T=T).to(args.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        # Train model
        losses = train(model, optimizer, latent_loader, args.epochs, args.device)

        # Plot training loss
        import matplotlib.pyplot as plt
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig("training_loss.png")
        plt.close()

        # Save model
        torch.save(model.state_dict(), args.model)
    
    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        D = 64
        # network = FcNetwork(D, num_hidden)
        network = LatentUnet()

        # Set the number of steps in the diffusion process
        T = 1000

        # Define model
        model = DDPM(network, T=T).to(args.device)

        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample((4, 1, 8, 8))).cpu() 
            # Denormalize samples using the saved mean and std from training
            stats = torch.load(args.latent_stats)
            samples = samples * stats['z_std'] + stats['z_mean']
        # Samples must be reshaped to (4, 64) before passing to decoder network
        samples = samples.view(samples.shape[0], -1) # Reshape to (4, 64)

        # Pass samples to decoder to get images
        # Loading VAE model
        M = args.latent_dim
        if args.prior == 'gaussian':
            prior = GaussianPrior(M)

        # Define encoder and decoder networks
        encoder_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, M*2),
        )

        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Unflatten(-1, (28, 28))
        )

        # Define VAE model
        encoder = GaussianEncoder(encoder_net)
        if args.mnist_type == 'binarized':
            decoder = BernoulliDecoder(decoder_net)
        else:            
            decoder = GaussianDecoder(decoder_net)

        vae_model = VAE(prior, decoder, encoder, beta=args.beta).to(args.device)
        vae_model.load_state_dict(torch.load(args.vae_model, map_location=torch.device(args.device)))

        vae_model.eval()
        with torch.no_grad():
            samples = vae_model.decoder(samples.to(args.device)).sample().cpu() # Shape: (4, 28, 28)
            mean_samples = vae_model.decoder.mu.cpu() # Shape: (4, 28, 28)

        samples = samples.clamp(0, 1)

        # Plot first samples
        fig, axes = plt.subplots(1, 4, figsize=(5, 5))
        for i in range(4):
            axes[i].imshow(samples[i].squeeze(), cmap='gray')
            axes[i].axis('off')
        plt.savefig(args.samples)
        plt.close()

        # Plot mean samples
        fig, axes = plt.subplots(1, 4, figsize=(5, 5))
        for i in range(4):
            axes[i].imshow(mean_samples[i].squeeze(), cmap='gray')
            axes[i].axis('off')
        plt.savefig("mean_" + args.samples)
        plt.close()

    elif args.mode == 'speed':
        def _sync(device):
            if device.type == "cuda": torch.cuda.synchronize()
            elif device.type == "mps":
                try: torch.mps.synchronize()
                except: pass

        @torch.no_grad()
        def measure_sps(sample_fn, total_samples, batch_size, device, warmup=5):
            for _ in range(warmup):
                _ = sample_fn(batch_size)

            _sync(device)
            n, t0 = 0, time.perf_counter()
            with tqdm(total = total_samples, desc="Measuring speed") as pbar:
                while n < total_samples:
                    b = min(batch_size, total_samples - n)
                    _ = sample_fn(b)
                    n += b
                    pbar.update(b)

            _sync(device)
            t1 = time.perf_counter()

            return total_samples/(t1-t0), (t1-t0)

        device = torch.device(args.device)

        network = LatentUnet()
        model = DDPM(network, T=1000).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()

        M = args.latent_dim
        prior = GaussianPrior(M)
        encoder_net = nn.Sequential(
            nn.Flatten(), nn.Linear(784,512), nn.ReLU(),
            nn.Linear(512,512), nn.ReLU(),
            nn.Linear(512, M*2),
        )
        decoder_net = nn.Sequential(
            nn.Linear(M,512), nn.ReLU(),
            nn.Linear(512,512), nn.ReLU(),
            nn.Linear(512,784), nn.Unflatten(-1,(28,28))
        )
        encoder = GaussianEncoder(encoder_net)
        decoder = GaussianDecoder(decoder_net)  
        vae_model = VAE(prior, decoder, encoder, beta=args.beta).to(device)
        vae_model.load_state_dict(torch.load(args.vae_model, map_location=device))
        vae_model.eval()

        def latent_ddpm_sample_fn(b):
            z = model.sample((b, 1, 8, 8))   
            
            stats = torch.load('latent_stats.pt')
            z = z * stats['z_std'] + stats['z_mean']  
            z = z.view(b, -1)                  
            x = vae_model.decoder(z).sample()  
            return x

        sps, elapsed = measure_sps(latent_ddpm_sample_fn, args.speed_total, args.speed_batch, device)
        print(f"Latent DDPM sampling (incl decode): {sps:.2f} samples/sec (elapsed {elapsed:.2f}s)")