import torch
from typing import List

from models.embeddings.embedding import Embedding
from utils import binary_rounding


class Embedding_v1(Embedding):
    def __init__(
        self,
        device: torch.device,
        all_embeddings: List[List[str]],
        embedding_dim: int = 256,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        # Device
        self.device = device

        # Total dimension of embeddings
        self.embedding_dim = embedding_dim

        # Single field dimensions
        self.single_embedding_dim = embedding_dim // binary_rounding(
            len(all_embeddings)
        )

        # Convert List[List[str]] of embeddings into List[Dict]
        __list_set = [set(inner_list) for inner_list in all_embeddings]
        self.embedding_dicts = [
            {element: index for index, element in enumerate(tup)} for tup in __list_set
        ]

        # List of embedders
        self.embedders = [
            torch.nn.Embedding(len(embedding_dict), self.single_embedding_dim).to(
                self.device
            )
            for embedding_dict in self.embedding_dicts
        ]

    def __call__(self, *args, **kwargs):
        pass

    def step_embedding(self, step):
        """
        This function implement Sinusoidal positional embeddings.
        Which generates embeddings using sin and cos functions
        Input: tensor shape (N)
        :return: embedding tensor shape of (N, self.embedding_dims)
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
