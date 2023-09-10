from entities.data.wound_original import WoundOriginal
from models.trainer.vae_trainer import VAETrainer

if __name__ == '__main__':
    dataset = WoundOriginal()
    vae_trainer = VAETrainer(
        train_dataset=dataset
    )
    vae_trainer.fit()
    # data = dataset.data
    # print(data)
