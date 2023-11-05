import CONST
from entities import WoundDataset
from models.trainer.diffuser import Diffuser

if __name__ == '__main__':
    # The data
    dataset = WoundDataset(
        image_dir=CONST.PROCESSED_IMAGES_DIR,
        segment_dir=CONST.PROCESSED_SEGMENT_DIR,
        target_tensor_size=CONST.DIFFUSER_SETTINGS.INPUT_SIZE,
        embedding_dir=CONST.PROCESSED_EMBEDDING_DIR
    )

    diffuser = Diffuser(
        dataset=dataset,
        batch_size=24,
        num_workers=2,
        run_name=CONST.DIFFUSER_SETTINGS.RUN_NAME,
        output_dir=CONST.DIFFUSER_SETTINGS.OUTPUT_DIR,
        attn_heads=2,
    )

    diffuser.fit()


