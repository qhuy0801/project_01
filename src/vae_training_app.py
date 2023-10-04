import gc

import CONST
from entities import WoundDataset
from models.nets.vae_v4 import VAE_v4
from models.trainer.vae_trainer import VAETrainer

if __name__ == '__main__':
    # Initialise the model
    model = VAE_v4(
        input_size=CONST.VAE_SETTING_v4.INPUT_SIZE,
    )

    # # Initialise dataset
    dataset = WoundDataset(
        image_dir=CONST.PROCESSED_IMAGES_DIR,
        segment_dir=CONST.PROCESSED_SEGMENT_DIR,
        annotation_path=CONST.ANNOTATION_PROCESSED_PATH,
        target_tensor_size=CONST.VAE_SETTING_v4.INPUT_SIZE,
    )

    # Create the training application
    vae_trainer = VAETrainer(
        train_dataset=dataset,
        model=model,
        batch_size=CONST.VAE_SETTING_v4.BATCH_SIZE,
        checkpoint_path=CONST.VAE_SETTING_v4.CHECKPOINT_PATH,
        num_workers=CONST.VAE_SETTING_v4.NUM_WORKERS,
        num_samples=CONST.VAE_SETTING_v4.NUM_SAMPLES,
        epochs=CONST.VAE_SETTING_v4.EPOCHS,
        max_lr=CONST.VAE_SETTING_v4.MAX_LR,
        min_lr=CONST.VAE_SETTING_v4.MIN_LR,
        patience_lr=CONST.VAE_SETTING_v4.PATIENCE_LR,
        output_dir=CONST.OUTPUT_DIR,
        run_name=CONST.VAE_SETTING_v4.RUN_NAME,
    )

    # As we created extra instances, we will need to un-referent them before training
    model = None
    dataset = None
    gc.collect()

    # Training trigger
    vae_trainer.fit()
