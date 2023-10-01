import gc

import bitsandbytes as bnb
from torchinfo import summary

import CONST
from entities import WoundDataset
from models.nets.vae_v3 import VAE_v3
from models.nets.vae_v4 import Autoencoder_v1
from models.trainer.ae_trainer import AETrainer_v1
from models.trainer.vae_trainer import VAETrainer
from models.trainer.vae_trainer_v2 import VAETrainer_v2

if __name__ == '__main__':
    # Initialise the model
    model = Autoencoder_v1(
        input_size=CONST.AE_SETTING_v1.INPUT_SIZE,
    )

    # # Initialise dataset
    dataset = WoundDataset(
        image_dir=CONST.PROCESSED_IMAGES_DIR,
        segment_dir=CONST.PROCESSED_SEGMENT_DIR,
        annotation_path=CONST.ANNOTATION_PROCESSED_PATH,
        target_tensor_size=CONST.VAE_SETTING_v3.INPUT_SIZE,
    )

    # Create the training application
    vae_trainer = AETrainer_v1(
        train_dataset=dataset,
        model=model,
        batch_size=CONST.AE_SETTING_v1.BATCH_SIZE,
        checkpoint_path=CONST.AE_SETTING_v1.CHECKPOINT_PATH,
        num_workers=CONST.AE_SETTING_v1.NUM_WORKERS,
        num_samples=CONST.AE_SETTING_v1.NUM_SAMPLES,
        epochs=CONST.AE_SETTING_v1.EPOCHS,
        max_lr=CONST.AE_SETTING_v1.MAX_LR,
        output_dir=CONST.OUTPUT_DIR,
        run_name=CONST.AE_SETTING_v1.RUN_NAME,
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
