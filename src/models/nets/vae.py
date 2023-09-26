from abc import ABC, abstractmethod

from torch import nn


class VAE(ABC, nn.Module):

    @abstractmethod
    def encode(self, x):
        """Encode the input into a latent representation."""
        raise NotImplementedError("The 'encode' method must be overridden by subclasses")

    @abstractmethod
    def decode(self, z):
        """Decode the latent representation into the original pixel space."""
        raise NotImplementedError("The 'decode' method must be overridden by subclasses")

    @abstractmethod
    def forward(self, x):
        """Forward pass through the VAE (encode then decode)."""
        raise NotImplementedError("The 'forward' method must be overridden by subclasses")
