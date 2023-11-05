import wandb

import CONST
from entities import WoundDataset
from models.trainer.diffuser import Diffuser


# Define the agent
# Run this on agent computers
def train():
    """
    Training trigger for hyper-parameter optimisation
    :return: None
    """
    # Init the run
    run = wandb.init(project="DDPM_hyper_tuning")

    # Get the configuration for instance
    config = wandb.config

    # Similar approach to full train
    # The data
    dataset = WoundDataset(
        image_dir=CONST.PROCESSED_IMAGES_DIR,
        segment_dir=CONST.PROCESSED_SEGMENT_DIR,
        target_tensor_size=CONST.DIFFUSER_SETTINGS.INPUT_SIZE,
        embedding_dir=CONST.PROCESSED_EMBEDDING_DIR
    )

    diffuser = Diffuser(
        dataset=dataset,
        batch_size=8,
        num_workers=2,
        epochs=500,
        run_name=CONST.DIFFUSER_SETTINGS.RUN_NAME,
        output_dir=CONST.DIFFUSER_SETTINGS.OUTPUT_DIR,
        max_lr=config.learning_rate,
        noise_steps=config.noise_steps,
        variance_schedule_type=config.variance_schedule_type,
        attn_heads=config.attn_heads,
        wandb_run=run,
    )

    diffuser.fit()


if __name__ == '__main__':
    # Login wandb
    wandb_key = "a8b5a7676a58d9b5b1e686fd9d349bc25f18d07c"
    wand_logged = wandb.login(key=wandb_key)

    # Start the sweep agent
    wandb.agent(sweep_id="2i9tp395", project="DDPM_hyper_tuning", function=train, count=10)
