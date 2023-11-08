import wandb

import CONST
from entities import WoundDataset
from models.trainer.diffuser import Diffuser

if __name__ == "__main__":
    # Login wandb
    wandb_key = "a8b5a7676a58d9b5b1e686fd9d349bc25f18d07c"
    wand_logged = wandb.login(key=wandb_key)

    # Init the run
    run = wandb.init(
        project="DDPM_full_train",
        config={
            "variance_schedule_type": "cosine",
            "attn_heads": 1,
            "batch_size": 28,
            "num_workers": 2,
            "epochs": 10000,
            "max_lr": 1e-4,
            "noise_steps": 100,
        },
    )

    dataset = WoundDataset(
        image_dir=CONST.PROCESSED_IMAGES_DIR,
        segment_dir=CONST.PROCESSED_SEGMENT_DIR,
        target_tensor_size=CONST.DIFFUSER_SETTINGS.INPUT_SIZE,
        embedding_dir=CONST.PROCESSED_EMBEDDING_DIR,
    )
    diffuser = Diffuser(
        dataset=dataset,
        batch_size=28,
        num_workers=2,
        epochs=10000,
        run_name=CONST.DIFFUSER_SETTINGS.RUN_NAME,
        output_dir=CONST.DIFFUSER_SETTINGS.OUTPUT_DIR,
        max_lr=2e-4,
        noise_steps=100,
        variance_schedule_type="cosine",
        attn_heads=1,
        wandb_run=run,
    )

    # Trigger the training
    diffuser.fit()
