import gc

import CONST
from entities import WoundDataset
from models.nets.vae_v1 import VAE_v1
from models.trainer.vae_trainer import VAETrainer

if __name__ == '__main__':
    # Initialise the model
    model = VAE_v1(
        input_size=CONST.VAE_SETTING.INPUT_SIZE,
        dims=CONST.VAE_SETTING.DIM_CONFIG,
        latent_dim=CONST.VAE_SETTING.LATENT_DIM,
    )

    # Initialise dataset
    dataset = WoundDataset(
        image_dir=CONST.PROCESSED_IMAGES_DIR,
        segment_dir=CONST.PROCESSED_SEGMENT_DIR,
        annotation_path=CONST.ANNOTATION_PROCESSED_PATH,
        target_tensor_size=CONST.VAE_SETTING.INPUT_SIZE,
    )

    # Create the training application
    vae_trainer = VAETrainer(
        train_dataset=dataset,
        model=model,
        batch_size=CONST.VAE_SETTING.BATCH_SIZE,
        num_workers=CONST.VAE_SETTING.NUM_WORKERS,
        num_samples=CONST.VAE_SETTING.NUM_SAMPLES,
        epochs=CONST.VAE_SETTING.EPOCHS,
        max_lr=CONST.VAE_SETTING.MAX_LR
    )

    # As we created extra instances, we will need to un-referent them before training
    model = None
    dataset = None
    gc.collect()

    # Training trigger
    vae_trainer.fit()
