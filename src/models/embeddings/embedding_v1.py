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

    def __call__(self, steps, semantics):
        """
        Function to call embedding based on steps and semantics, with batch N
        :param steps: tensor of (N)
        :param semantics: list of N semantics
        :return: sumarisation of steps and semantics
        """
        return self.step_embeddings(steps) + self.semantic_embeddings(semantics)

    def step_embeddings(self, steps):
        """
        This function implement Sinusoidal positional embeddings.
        Which generates embeddings using sin and cos functions
        Input: tensor shape (N)
        :return: embedding tensor shape of (N, self.embedding_dims)
        """
        steps = steps.unsqueeze(-1).to(self.device)
        normalising_factor = 1.0 / (
            10000
            ** (
                torch.arange(0, self.embedding_dim, 2, device=self.device).float()
                / self.embedding_dim
            )
        )
        position_a = torch.sin(
            steps.repeat(1, self.embedding_dim // 2) * normalising_factor
        )
        position_b = torch.cos(
            steps.repeat(1, self.embedding_dim // 2) * normalising_factor
        )
        return torch.cat([position_a, position_b], dim=-1)

    def semantic_embeddings(self, semantics):
        """
        A functions that concat single embeddings into a batch
        :param semantics:
        :return:
        """
        embeddings = []
        for semantic in semantics:
            embedding = self.single_semantic_embedding(semantic)
            embeddings.append(embedding)
        return torch.cat(embeddings, dim=0)

    def single_semantic_embedding(self, semantic: List[List[str]]):
        """
        Get the requested semantic and returned embeddings based on it
        :param semantic:
        [["cherry", "tomato"], -> self.embedders[0]
         ["car", "bike", "subway"]] -> self.embedders[1]
        :return: tensor shaped (self.embedding_dim)
        """
        if len(semantic) != len(self.embedding_dicts):
            raise ValueError(
                "The category count is not matched to initialised embedders count."
            )

        # Filter duplicated by turning it into a set
        semantic = [set(inner_list) for inner_list in semantic]

        # Look up in dictionary and then turn to class
        # Initialize an empty list to store the modified sets
        semantic_classed = []

        # Iterate through each set and its corresponding dictionary in parallel
        for set_, dict_ in zip(semantic, self.embedding_dicts):
            # Initialize an empty set to store the modified elements
            modified_set = set()
            # Iterate through each element in the set
            for element in set_:
                # Look up the element in the dictionary and add the corresponding value to the modified set
                modified_set.add(dict_.get(element, element))
            # Append the modified set to the modified list of sets
            semantic_classed.append(modified_set)

        # Push elements through embedders (same approach as above)
        semantic_embedded = []
        for set_, embedder_ in zip(semantic_classed, self.embedders):
            modified_set = set()
            for element in set_:
                modified_set.add(
                    embedder_(
                        torch.tensor(float(element), dtype=torch.int).to(self.device)
                    )
                )
            semantic_embedded.append(modified_set)

        # Algorithm of embeddings
        semantic_category = []
        for category_ in semantic_embedded:
            semantic_category.append(sum(category_))

        if self.single_embedding_dim * len(self.embedding_dicts) == self.embedding_dim:
            return torch.cat(semantic_category, dim=0).unsqueeze(0)
        else:
            return torch.nn.functional.pad(
                torch.cat(semantic_category, dim=0),
                (
                    0,
                    self.embedding_dim
                    - self.single_embedding_dim * len(self.embedding_dicts),
                ),
                "constant",
                0,
            ).unsqueeze(0)
