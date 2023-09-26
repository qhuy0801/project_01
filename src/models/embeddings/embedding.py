from abc import ABC, abstractmethod

from torch import nn


class Embedding(ABC, nn.Module):
    """
    Abstract class representing an Embedding. This class defines the interface
    for embedding with abstract methods for step embedding, semantic embedding,
    combined embedding, and a call method. It also defines an abstract property
    for embedding dimension.
    """

    @abstractmethod
    def step_embeddings(self, *args, **kwargs):
        """
        Turn the step into an array of 1-D tensor.
        """
        raise NotImplementedError("Subclasses must implement this abstract method")

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Abstract method for calling the embedding instance.
        """
        raise NotImplementedError("Subclasses must implement this abstract method")
