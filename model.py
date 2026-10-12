import torch
import torch.nn.functional as F
import numpy as np


class MultiVAE(torch.nn.Module):
    """
    Multilayer Variational AutoEncoder (MultiVAE) for collaborative filtering.
    Reference: "Variational Autoencoders for Collaborative Filtering" (WWW 2018).

    Args:
        p_dims: List of dimensions for the encoder and decoder layers.
        dropout_p: Dropout probability.
    """

    def __init__(self, p_dims, dropout_p=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        self.q_dims = p_dims[::-1]  # Encoder dims are the reverse of decoder dims

        # The last layer of the encoder outputs both mean (mu) and log variance (logvar)
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]

        # Encoder layers
        self.q_layers = torch.nn.ModuleList([torch.nn.Linear(d_in, d_out) for
                                             d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])

        # Decoder layers
        self.p_layers = torch.nn.ModuleList([torch.nn.Linear(d_in, d_out) for
                                             d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        self.drop = torch.nn.Dropout(dropout_p)
        self.init_weights()

    def forward(self, input_data):
        """
        Forward pass of the VAE.

        Args:
            input_data: User interaction vector.

        Returns:
            recon_x: Reconstructed output.
            mu: Latent mean.
            logvar: Latent log variance.
        """
        mu, logvar = self.encoder(input_data)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def encoder(self, input_data):
        """Encoder: Maps input to latent space."""
        h = F.normalize(input_data, p=2, dim=1)
        h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample z."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def decoder(self, z):
        """Decoder: Maps latent code back to data space."""
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def init_weights(self):
        """Initialize network weights using Xavier Normal initialization."""
        for layer in self.q_layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.normal_(layer.bias, 0, 0.001)

        for layer in self.p_layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.normal_(layer.bias, 0, 0.001)

    def loss_function(self, recon_x, x, mu, logvar, anneal=1.0):
        """
        Compute VAE loss: Reconstruction Error + KL Divergence.

        Args:
            recon_x: Reconstructed input.
            x: Original input.
            mu: Latent mean.
            logvar: Latent log variance.
            anneal: KL annealing coefficient (Beta).

        Returns:
            Total loss.
        """
        # Reconstruction loss (Neg Log-Likelihood for Multinomial)
        neg_ll = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))

        # KL divergence
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        return neg_ll + anneal * KLD