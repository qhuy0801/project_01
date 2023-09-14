import torch
import bitsandbytes as bnb
from torch import optim

from entities.data.wound_original import WoundOriginal
from models.trainer.vae_trainer import VAETrainer

if __name__ == '__main__':
    dataset = WoundOriginal(
        target_size=256
    )
    vae_trainer = VAETrainer(
        train_dataset=dataset,
        batch_size=16,
        epochs=7000,
        max_lr=1e-4,
        num_workers=30
    )
    vae_trainer.fit()
