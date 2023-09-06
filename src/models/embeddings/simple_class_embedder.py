from abc import ABC

import torch
from torch import nn

from models import Embedder


class ClassEmbedder(Embedder, ABC):
    """
    A basic embedder for vectorisation based on categories (classes).

    Example:
        "A", "B", "C"
        1., 2., 3.,
    """

    def __init__(self, class_count, embedded_dim=256) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedded_dim = embedded_dim
        self.class_count = class_count
        self.class_embedding = nn.Embedding(self.class_count, self.embedded_dim).to(self.device)

    def step_embedding(self, step):
        """
        Take step input and return step embedded matrix
        :return:
        """
        step = step.unsqueeze(-1).to(self.device)
        normalising_factor = 1.0 / (
            10000
            ** (
                torch.arange(0, self.embedded_dim, 2, device=self.device).float()
                / self.embedded_dim
            )
        )
        position_a = torch.sin(
            step.repeat(1, self.embedded_dim // 2) * normalising_factor
        )
        position_b = torch.cos(
            step.repeat(1, self.embedded_dim // 2) * normalising_factor
        )
        return torch.cat([position_a, position_b], dim=-1)

    def semantic_embedding(self, semantic):
        """
        Take the class and return embedded matrix
        :return:
        """
        return self.class_embedding(semantic)

    def combine_embedding(self, step_embedding, semantic_embedding):
        """
        Combine function
        :return:
        """
        return step_embedding + semantic_embedding
