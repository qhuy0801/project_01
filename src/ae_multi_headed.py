import CONST
from entities import WoundDataset
from models.nets.vae_v4 import Multi_headed_AE
from models.trainer.multi_headed_ae_trainer import MultiheadAETrainer

if __name__ == '__main__':
    model = Multi_headed_AE()

    # # Initialise dataset
    dataset = WoundDataset(
        image_dir=CONST.PROCESSED_IMAGES_DIR,
        segment_dir=CONST.PROCESSED_SEGMENT_DIR,
        annotation_path=CONST.ANNOTATION_PROCESSED_PATH,
        target_tensor_size=CONST.AE_SETTING_v1.INPUT_SIZE,
    )

    multi_headed_ae_trainer = MultiheadAETrainer(
        train_dataset=dataset,
        model=model,
        batch_size=CONST.MULTI_HEADED_AE_SETTING.BATCH_SIZE,
        num_workers=CONST.MULTI_HEADED_AE_SETTING.NUM_WORKERS,
        num_samples=CONST.MULTI_HEADED_AE_SETTING.NUM_SAMPLES,
        epochs=CONST.MULTI_HEADED_AE_SETTING.EPOCHS,
        max_lr=CONST.MULTI_HEADED_AE_SETTING.MAX_LR,
        min_lr=CONST.MULTI_HEADED_AE_SETTING.MIN_LR,
        lr_decay=CONST.MULTI_HEADED_AE_SETTING.DECAY_RATE,
        lr_threshold=CONST.MULTI_HEADED_AE_SETTING.LR_THRESHOLD,
        patience_lr=CONST.MULTI_HEADED_AE_SETTING.PATIENCE_LR,
        max_lr_additional=CONST.MULTI_HEADED_AE_SETTING.ADDITIONAL_MAX_LR,
        min_lr_additional=CONST.MULTI_HEADED_AE_SETTING.ADDITIONAL_MIN_LR,
        lr_decay_additional=CONST.MULTI_HEADED_AE_SETTING.ADDITIONAL_DECAY_RATE,
        lr_threshold_additional=CONST.MULTI_HEADED_AE_SETTING.ADDITIONAL_LR_THRESHOLD,
        patience_lr_additional=CONST.MULTI_HEADED_AE_SETTING.ADDITIONAL_LR_PATIENCE,
        run_name=CONST.MULTI_HEADED_AE_SETTING.RUN_NAME,
        output_dir=CONST.MULTI_HEADED_AE_SETTING.OUTPUT_DIR,
        simple_ae_checkpoint=CONST.MULTI_HEADED_AE_SETTING.AE_CHECKPOINT,
    )

    model = None
    dataset = None

    multi_headed_ae_trainer.fit()
