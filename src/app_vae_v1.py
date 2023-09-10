from entities.data.wound_roi import WoundROI
from models.trainer.vae_trainer import VAETrainer

if __name__ == '__main__':
    wound_roi = WoundROI()
    vae_trainer = VAETrainer(
        train_dataset=wound_roi
    )
    vae_trainer.fit()
