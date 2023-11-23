import CONST
from entities import WoundDataset
from model_assembly.result_generator import Generator

if __name__ == '__main__':
    dataset = WoundDataset(
        image_dir=CONST.PROCESSED_IMAGES_DIR,
        segment_dir=CONST.PROCESSED_SEGMENT_DIR,
        target_tensor_size=CONST.DIFFUSER_SETTINGS.INPUT_SIZE,
        embedding_dir=CONST.PROCESSED_EMBEDDING_DIR,
        generation_mode=True,
    )

    generator = Generator(
        dataset=dataset,
        ddpm_checkpoint="../resources/checkpoint/ddpm_v1.pt"
    )

    generator.ddpm_generate_all(result_dir="../resources/output/ddpm_64/")
