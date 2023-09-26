import torch

from models.embeddings.embedding_v1 import Embedding_v1

if __name__ == "__main__":
    embedding = Embedding_v1(
        device=torch.device("cpu"),
        all_embeddings=[
            ["cherry", "tomato", "cherry"],
            ["cars", "bicycle"],
            ["swim", "batminton", "fising"],
        ],
    )

    print(embedding)
