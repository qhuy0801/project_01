from abc import ABC

import torch
from torch import nn


class VAE(ABC, nn.Module):

    def encode(self, x):
        """
        Encode the input
        :param x: Tensor[N, C, H, W]
        :return: Tensor[N, latent_dim], Tensor[N, latent_dim]
        """
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc_mu(x), self.fc_sigma(x)

    def decode(self, x):
        """
        Decode from latent space into pixel space
        :param x:
        :return:
        """
        x = self.decoder_input(x)
        x = x.view(
            -1, self.dims[-1][-1], self.compressed_conv_size, self.compressed_conv_size
        )
        return self.decoder(x)

    def reparameterise(self, mu, sigma):
        """
        Sample from mu and sigma: mu + std + eps
        :param mu:
        :param sigma:
        :return:
        """
        std = torch.exp(sigma * 0.5)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        """
        Forward function of VAE: encode > reparameterise > decode
        :param x:
        :return:
        """
        mu, sigma = self.encode(x)
        x = self.reparameterise(mu, sigma)
        return self.decode(x), mu, sigma
