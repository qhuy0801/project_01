import torch
import bitsandbytes as bnb

from entities.data.wound_original import WoundOriginal
from models.trainer.vae_trainer import VAETrainer

if __name__ == '__main__':
    dataset = WoundOriginal(
        target_size=256
    )
    # vae_trainer = VAETrainer(
    #     train_dataset=dataset
    # )
    # vae_trainer.fit()

    # We will do retrain from the checkpoint with a new learning rate scheduler
    vae_trainer = VAETrainer(
        train_dataset=dataset,
        checkpoint_path="./output/vae_v2/09130038/vae_v2.pt",
        batch_size=64,
        epochs=5000
    )

    # Get a new optimiser and learning rate scheduler
    vae_trainer.optimiser = bnb.optim.AdamW(
        vae_trainer.model.parameters(),
        lr=1e-3
    )
    vae_trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=vae_trainer.optimiser,
        step_size=len(vae_trainer.train_data),
        gamma=0.999,
    )
    vae_trainer.fit()
