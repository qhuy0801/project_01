from abc import ABC, abstractmethod


class Embedder(ABC):
    """
    Abstract class for embedding integration.

    This class is designed to ensure the adoption of the Inversion of Control (IoC) and
    Dependency Injection (DI) principles to facilitate the integration of later upgraded
    embedding method.
    """
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """
        Constructor
        :param args:
        :param kwargs:
        """
        super().__init__()

    @abstractmethod
    def step_embedding(self, *args, **kwargs):
        """
        Retrieve the embedding matrix corresponding to the current noise step.

        This method returns the embedding matrix that represents the current
        step in the noise diffusion process.

        :return: numpy.ndarray: The embedding matrix for the current noise step.
        """

    @abstractmethod
    def semantic_embedding(self, *args, **kwargs):
        """
        Retrieve the embedding matrix representing the semantic information
        of the object intended for rendering.

        This method returns the embedding matrix that encapsulates the semantic
        details of the target object, facilitating its rendering process.

        :return: numpy.ndarray: The embedding matrix encapsulating the semantic
        information of the desired object.
        """

    @abstractmethod
    def combine_embedding(self, *args, **kwargs):
        """
        Combine the semantic and timestep embeddings to produce a unified
        embedding for subsequent training.

        This method merges the provided semantic and timestep embeddings to
        generate a single cohesive embedding. This unified embedding captures
        both the semantic information of the object and its associated timestep,
        making it suitable for further training processes.
        :return: numpy.ndarray: The combined unified embedding.
        """
