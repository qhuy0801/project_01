import wandb

from models.trainer.upscaler_trainer import UpscalerTrainer
import CONST
from entities import WoundDataset


def train():
    # Init the run
    run = wandb.init(project="Up_scaler")

    # Get the configuration for instance
    config = wandb.config

    # Training setups
    dataset = WoundDataset(
        image_dir=CONST.PROCESSED_IMAGES_DIR,
        segment_dir=CONST.PROCESSED_SEGMENT_DIR,
        target_tensor_size=CONST.DIFFUSER_SETTINGS.INPUT_SIZE,
        additional_target_tensor_size=256,
    )

    decoder_trainer = UpscalerTrainer(
        dataset=dataset,
        max_lr=1e-4,
        epochs=500,
        output_dir="../resources/output/",
        hidden_channels=config.hidden_channels,
        middle_activation=config.middle_activation,
        output_module=config.output_module,
        wandb_run=run,
    )

    # Trigger the training
    decoder_trainer.fit()


if __name__ == "__main__":
    # Login wandb
    wandb_key = "a8b5a7676a58d9b5b1e686fd9d349bc25f18d07c"
    wand_logged = wandb.login(key=wandb_key)

    # Start the sweep agent
    wandb.agent(sweep_id="94374qrj", project="Up_scaler", function=train, count=16)
