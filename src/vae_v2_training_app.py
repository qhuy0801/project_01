import gc
import torch

import CONST
from entities import WoundDataset
from models.nets.vae_v2 import VAE_v2
from models.trainer.vae_trainer import VAETrainer

if __name__ == '__main__':
    # Initialise the model
    model = VAE_v2(
        input_size=CONST.VAE_SETTING.INPUT_SIZE,
        dims=CONST.VAE_SETTING.DIM_CONFIG,
        latent_dim=CONST.VAE_SETTING.LATENT_DIM,
    )

    # print(summary(model, (3, 256, 256)))

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
        checkpoint_path=CONST.VAE_SETTING.CHECKPOINT_PATH,
        num_workers=CONST.VAE_SETTING.NUM_WORKERS,
        num_samples=CONST.VAE_SETTING.NUM_SAMPLES,
        epochs=CONST.VAE_SETTING.EPOCHS,
        max_lr=CONST.VAE_SETTING.MAX_LR,
        lr_decay=CONST.VAE_SETTING.DECAY_RATE,
        output_dir=CONST.OUTPUT_DIR,
        run_name=CONST.VAE_SETTING.RUN_NAME,
    )

    # Create new LR scheduler as we need to reset it
    vae_trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=vae_trainer.optimiser,
        step_size=len(vae_trainer.train_data),
        gamma=CONST.VAE_SETTING.DECAY_RATE,
        last_epoch=-1,
    )

    # Un-reference instances
    model = None
    gc.collect()

    # Training trigger
    vae_trainer.fit()
