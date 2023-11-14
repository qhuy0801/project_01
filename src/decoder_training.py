from models.trainer.decoder_trainer import DecoderTrainer
import CONST
from entities import WoundDataset
from models.trainer.diffuser import Diffuser

if __name__ == "__main__":
    dataset = WoundDataset(
        image_dir=CONST.PROCESSED_IMAGES_DIR,
        segment_dir=CONST.PROCESSED_SEGMENT_DIR,
        target_tensor_size=CONST.DIFFUSER_SETTINGS.INPUT_SIZE,
        additional_target_tensor_size=256,
    )

    decoder_trainer = DecoderTrainer(
        dataset=dataset,
        max_lr=1e-4,
    )

    # Trigger the training
    decoder_trainer.fit()