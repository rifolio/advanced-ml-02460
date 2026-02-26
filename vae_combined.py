# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.nn import functional as F
from tqdm import tqdm


# =============================================================================
# From vae_mog.py
# =============================================================================

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

    def log_prob(self, z):
        return self().log_prob(z)

    def sample(self, n_samples):
        return self().sample(torch.Size([n_samples]))


class MoGPrior(nn.Module):
    """
    Mixture of Gaussians prior for VAE.
    """
    def __init__(self, M, K=10, component_std=0.1):
        super(MoGPrior, self).__init__()
        self.M = M
        self.K = K
        self.component_std = component_std
        self.logits = nn.Parameter(torch.zeros(K))
        self.means = nn.Parameter(torch.randn(K, M) * 0.1)

    def forward(self):
        weights = torch.softmax(self.logits, dim=0)
        mixture = td.Categorical(probs=weights)
        components = td.Independent(
            td.Normal(loc=self.means, scale=torch.full_like(self.means, self.component_std)),
            1,
        )
        return MixtureSameFamily(mixture, components)

    def log_prob(self, z):
        return self().log_prob(z)

    def sample(self, n_samples):
        return self().sample(torch.Size([n_samples]))


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        std = torch.exp(std).clamp(min=1e-4, max=1e4)
        return td.Independent(td.Normal(loc=mean, scale=std), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        reconstruction = self.decoder(z).log_prob(x)
        log_qz = q.log_prob(z)
        log_pz = self.prior.log_prob(z)
        elbo = (reconstruction + log_pz - log_qz).mean()
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior.sample(n_samples)
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, train_loader, test_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    train_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    test_loader: [torch.utils.data.DataLoader]
            The data loader to use for evaluation.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.

    Returns:
    train_losses: [list of float]
        Mean training loss (negative ELBO) per epoch.
    test_losses: [list of float]
        Mean test loss (negative ELBO) per epoch.
    """
    train_losses = []
    test_losses = []

    total_steps = len(train_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_losses = []
        for x in train_loader:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

            progress_bar.set_postfix(
                train_loss=f"{loss.item():.4f}",
                epoch=f"{epoch+1}/{epochs}"
            )
            progress_bar.update()

        train_losses.append(sum(epoch_train_losses) / len(epoch_train_losses))

        # Evaluation on test set
        model.eval()
        epoch_test_losses = []
        with torch.no_grad():
            for x in test_loader:
                x = x[0].to(device)
                loss = model(x)
                epoch_test_losses.append(loss.item())

        test_losses.append(sum(epoch_test_losses) / len(epoch_test_losses))
        progress_bar.set_postfix(
            train_loss=f"{train_losses[-1]:.4f}",
            test_loss=f"{test_losses[-1]:.4f}",
            epoch=f"{epoch+1}/{epochs}"
        )

    return train_losses, test_losses


# =============================================================================
# From vaeflow.py
# =============================================================================

class GaussianBase(nn.Module):
    def __init__(self, D):
        """
        Define a Gaussian base distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the base distribution.
        """
        super(GaussianBase, self).__init__()
        self.D = D
        self.mean = nn.Parameter(torch.zeros(self.D), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.D), requires_grad=False)

    def forward(self):
        """
        Return the base distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class MaskedCouplingLayer(nn.Module):
    """
    An affine coupling layer for a normalizing flow.
    """

    def __init__(self, scale_net, translation_net, mask):
        """
        Define a coupling layer.

        Parameters:
        scale_net: [torch.nn.Module]
            The scaling network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        translation_net: [torch.nn.Module]
            The translation network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        mask: [torch.Tensor]
            A binary mask of dimension `(feature_dim,)` that determines which features (where the mask is zero) are transformed by the scaling and translation networks.
        """
        super(MaskedCouplingLayer, self).__init__()
        self.scale_net = scale_net
        self.translation_net = translation_net
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, z):
        """
        Transform a batch of data through the coupling layer (from the base to data).

        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations of dimension `(batch_size, feature_dim)`.
        """
        z_mask = self.mask * z
        s = self.scale_net(z_mask) * (1 - self.mask)
        s = torch.tanh(s)
        t = self.translation_net(z_mask) * (1 - self.mask)
        x = z_mask + (1 - self.mask) * (z * torch.exp(s) + t)
        log_det_J = s.sum(dim=1)

        return x, log_det_J
    
    def inverse(self, x):
        """
        Transform a batch of data through the coupling layer (from data to the base).

        Parameters:
        z: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        x: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        x_mask = self.mask * x
        s = self.scale_net(x_mask) * (1 - self.mask)
        s = torch.tanh(s)
        t = self.translation_net(x_mask) * (1 - self.mask)
        z = self.mask * x + (1 - self.mask) * ((x - t) * torch.exp(-s))
        log_det_J = (-s).sum(dim=1)

        return z, log_det_J


class Flow(nn.Module):
    def __init__(self, base, transformations):
        """
        Define a normalizing flow model.
        
        Parameters:
        base: [torch.distributions.Distribution]
            The base distribution.
        transformations: [list of torch.nn.Module]
            A list of transformations to apply to the base distribution.
        """
        super(Flow, self).__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)

    def forward(self, z):
        """
        Transform a batch of data through the flow (from the base to data).
        
        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations.            
        """
        sum_log_det_J = torch.zeros(z.size(0), device=z.device)
        for T in self.transformations:
            x, log_det_J = T(z)
            sum_log_det_J += log_det_J
            z = x
        return x, sum_log_det_J
    
    def inverse(self, x):
        """
        Transform a batch of data through the flow (from data to the base).

        Parameters:
        x: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        sum_log_det_J = torch.zeros(x.size(0), device=x.device)
        for T in reversed(self.transformations):
            z, log_det_J = T.inverse(x)
            sum_log_det_J += log_det_J
            x = z
        return z, sum_log_det_J
    
    def log_prob(self, x):
        """
        Compute the log probability of a batch of data under the flow.

        Parameters:
        x: [torch.Tensor]
            The data of dimension `(batch_size, feature_dim)`
        Returns:
        log_prob: [torch.Tensor]
            The log probability of the data under the flow.
        """
        z, log_det_J = self.inverse(x)
        return self.base().log_prob(z) + log_det_J
    
    def sample(self, sample_shape=(1,)):
        """
        Sample from the flow.

        Parameters:
        n_samples: [int]
            Number of samples to generate.
        Returns:
        z: [torch.Tensor]
            The samples of dimension `(n_samples, feature_dim)`
        """
        z = self.base().sample(sample_shape)
        return self.forward(z)[0]
    
    def loss(self, x):
        """
        Compute the negative mean log likelihood for the given data bath.

        Parameters:
        x: [torch.Tensor] 
            A tensor of dimension `(batch_size, feature_dim)`
        Returns:
        loss: [torch.Tensor]
            The negative mean log likelihood for the given data batch.
        """
        return -torch.mean(self.log_prob(x))

class FlowPrior(nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def log_prob(self, z):
        return self.flow.log_prob(z)

    def sample(self, n_samples):
        return self.flow.sample((n_samples,))


# =============================================================================
# Main - argument parsing and plots like vae_mog
# =============================================================================

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import matplotlib.pyplot as plt
    import glob
    import os

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--output-dir', type=str, default='outputs', help='base directory for models, samples, and plots (default: %(default)s)')
    parser.add_argument('--model', type=str, default=None, help='file to save model to or load model from (default: outputs/models/model_{prior}.pt)')
    parser.add_argument('--samples', type=str, default=None, help='file to save samples in (default: outputs/samples/samples_{prior}.png)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'flow'], help='prior distribution (default: %(default)s)')
    parser.add_argument('--plot-loss', type=str, default=None, help='file to save loss curve plot (default: outputs/plots/loss_curve_{prior}.png)')
    # flow prior settings
    parser.add_argument('--flow-steps', type=int, default=6)
    parser.add_argument('--flow-hidden', type=int, default=128)

    args = parser.parse_args()
    # Default paths: save into output_dir subfolders
    models_dir = os.path.join(args.output_dir, 'models')
    samples_dir = os.path.join(args.output_dir, 'samples')
    plots_dir = os.path.join(args.output_dir, 'plots')
    if args.model is None:
        args.model = os.path.join(models_dir, f'model_{args.prior}.pt')
    if args.plot_loss is None:
        args.plot_loss = os.path.join(plots_dir, f'loss_curve_{args.prior}.png')
    if args.samples is None:
        args.samples = os.path.join(samples_dir, f'samples_{args.prior}.png')
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define encoder and decoder networks (shared architecture)
    M = args.latent_dim
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

    # Define prior and model based on prior type
    if args.prior == 'gaussian':
        prior = GaussianPrior(M)
    elif args.prior == 'mog':
        prior = MoGPrior(M, K=10, component_std=0.1)
    elif args.prior == 'flow':
        base = GaussianBase(M)
        transformations = []
        mask = torch.zeros(M)
        mask[M//2:] = 1
        for _ in range(args.flow_steps):
            mask = 1 - mask
            scale_net = nn.Sequential(
                nn.Linear(M, args.flow_hidden), nn.ReLU(),
                nn.Linear(args.flow_hidden, M)
            )
            translation_net = nn.Sequential(
                nn.Linear(M, args.flow_hidden), nn.ReLU(),
                nn.Linear(args.flow_hidden, M)
            )
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
        flow = Flow(base, transformations)
        prior = FlowPrior(flow)

    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train_losses, test_losses = train(
            model, optimizer, mnist_train_loader, mnist_test_loader, args.epochs, args.device
        )

        # Save model
        model_dir = os.path.dirname(args.model)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), args.model)

        # Plot loss curve
        plot_dir = os.path.dirname(args.plot_loss)
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, 'b-', linewidth=2, label='Train')
        plt.plot(test_losses, 'r-', linewidth=2, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (negative ELBO)')
        plt.title(f'Train & Test Loss - {args.prior} prior')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.plot_loss)
        plt.close()
        print(f"Loss curve saved to {args.plot_loss}")

        # Generate and save samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu()
        samples_dir_path = os.path.dirname(args.samples)
        if samples_dir_path:
            os.makedirs(samples_dir_path, exist_ok=True)
        save_image(samples.view(64, 1, 28, 28), args.samples)
        print(f"Samples saved to {args.samples}")

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu()
        samples_dir_path = os.path.dirname(args.samples)
        if samples_dir_path:
            os.makedirs(samples_dir_path, exist_ok=True)
        save_image(samples.view(64, 1, 28, 28), args.samples)
