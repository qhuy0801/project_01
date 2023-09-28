import gc

import bitsandbytes as bnb
from torchinfo import summary

import CONST
from entities import WoundDataset
from models.nets.vae_v3 import VAE_v3
from models.trainer.vae_trainer import VAETrainer
from models.trainer.vae_trainer_v2 import VAETrainer_v2

if __name__ == '__main__':
    # Initialise the model
    model = VAE_v3(
        input_size=CONST.VAE_SETTING_v3.INPUT_SIZE,
        encoder_dim=CONST.VAE_SETTING_v3.ENCODER_DIM,
        decoder_dim=CONST.VAE_SETTING_v3.DECODER_DIM,
        latent_dim=CONST.VAE_SETTING_v3.LATENT_DIM,
    )

    # # Initialise dataset
    dataset = WoundDataset(
        image_dir=CONST.PROCESSED_IMAGES_DIR,
        segment_dir=CONST.PROCESSED_SEGMENT_DIR,
        annotation_path=CONST.ANNOTATION_PROCESSED_PATH,
        target_tensor_size=CONST.VAE_SETTING_v3.INPUT_SIZE,
        additional_target_tensor_size=CONST.VAE_SETTING_v3.ADDITIONAL_INPUT_SIZE,
    )

    # Create the training application
    vae_trainer = VAETrainer_v2(
        train_dataset=dataset,
        model=model,
        batch_size=CONST.VAE_SETTING_v3.BATCH_SIZE,
        checkpoint_path=CONST.VAE_SETTING_v3.CHECKPOINT_PATH,
        num_workers=CONST.VAE_SETTING_v3.NUM_WORKERS,
        num_samples=CONST.VAE_SETTING_v3.NUM_SAMPLES,
        epochs=CONST.VAE_SETTING_v3.EPOCHS,
        max_lr=CONST.VAE_SETTING_v3.MAX_LR,
        output_dir=CONST.OUTPUT_DIR,
        run_name=CONST.VAE_SETTING_v3.RUN_NAME,
    )
    #
    # # Re-train: get new optimiser and remove learning rate scheduler
    # # vae_trainer.optimiser = bnb.optim.AdamW(params=vae_trainer.model.parameters(), lr=CONST.VAE_SETTING.MAX_LR)
    #
    # # Remove learning rate scheduler
    # vae_trainer.lr_scheduler = None

    # As we created extra instances, we will need to un-referent them before training
    model = None
    dataset = None
    gc.collect()

    # Training trigger
    vae_trainer.fit()
