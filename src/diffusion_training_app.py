import torch

from models.embeddings.embedding_v1 import Embedding_v1

if __name__ == "__main__":
    embedding = Embedding_v1(
        device=torch.device("cpu"),
        all_embeddings=[
            ["cherry", "tomato", "cherry", "spinach"],
            ["cars", "bicycle"],
            ["swim", "batminton", "fishing"],
        ],
        embedding_dim=128,
    )

    timestep = embedding.step_embeddings(
        torch.tensor([20, 30, 40], dtype=torch.float32)
    )

    semantic = embedding.semantic_embeddings(
        [
            [["cherry", "spinach"], ["cars"], ["swim", "batminton", "fishing"]],
            [["spinach"], ["cars", "bicycle"], ["swim", "fishing"]],
            [["cherry", "tomato", "cherry", "spinach"], ["bicycle"], ["fishing"]],
        ]
    )

    embeddings_ = embedding(
        torch.tensor([20, 30, 40], dtype=torch.float32),
        [
            [["cherry", "spinach"], ["cars"], ["swim", "batminton", "fishing"]],
            [["spinach"], ["cars", "bicycle"], ["swim", "fishing"]],
            [["cherry", "tomato", "cherry", "spinach"], ["bicycle"], ["fishing"]],
        ],
    )
