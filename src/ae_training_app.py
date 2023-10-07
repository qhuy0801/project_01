import gc

import CONST
from entities import WoundDataset
from models.nets.vae_v4 import Autoencoder_v1
from models.trainer.ae_trainer import AETrainer_v1

if __name__ == '__main__':
    # Initialise the model
    model = Autoencoder_v1()

    # # Initialise dataset
    dataset = WoundDataset(
        image_dir=CONST.PROCESSED_IMAGES_DIR,
        segment_dir=CONST.PROCESSED_SEGMENT_DIR,
        annotation_path=CONST.ANNOTATION_PROCESSED_PATH,
        target_tensor_size=CONST.AE_SETTING_v1.INPUT_SIZE,
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
        output_dir=CONST.AE_SETTING_v1.OUTPUT_DIR,
        run_name=CONST.AE_SETTING_v1.RUN_NAME,
        lr_decay=CONST.AE_SETTING_v1.LR_DECAY,
        min_lr=CONST.AE_SETTING_v1.MIN_LR,
        lr_threshold=CONST.AE_SETTING_v1.LR_THRESHOLD,
        lr_reducing_patience=CONST.AE_SETTING_v1.PATIENCE_LR,
    )

    model = None
    dataset = None
    gc.collect()

    # Training trigger
    vae_trainer.fit()
