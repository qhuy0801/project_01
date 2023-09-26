import torch

from models.embeddings.embedding import Embedding


class Embedding_v1(Embedding):

    def __init__(self, embedding_dim: int = 256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim

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

