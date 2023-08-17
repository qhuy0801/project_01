from types import SimpleNamespace

from torch.utils.data import DataLoader

from entities.WoundDataset import WoundDataset
import constraints.SEG_CONST as SEG_CONST
from models.generation.generation_v1.diffuser.Diffuser import Diffuser

config = SimpleNamespace(
    run_name="DDPM_conditional",
    epochs=100,
    noise_steps=1000,
    seed=42,
    batch_size=10,
    img_size=512,
    num_classes=6,
    train_folder="train",
    val_folder="test",
    device="cpu",
    slice_size=1,
    do_validation=True,
    fp16=True,
    log_every_epoch=10,
    num_workers=10,
    learning_rate=5e-3,
)

if __name__ == "__main__":
    wound_dataset = WoundDataset(SEG_CONST.SEGMENTED_DIR)
    dataloader = DataLoader(wound_dataset, batch_size=10)

    diffuser = Diffuser(
        _train_data=dataloader,
        _img_size=512,
        _device="cpu",
        _beta_start=1e-4,
        _beta_end=0.02,
        _noise_step=1000,
        _class_count=6,
    )
    diffuser.setting_up(config)
    diffuser.fit(config)
