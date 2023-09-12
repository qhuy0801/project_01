import os
from datetime import datetime
from typing import Optional

import torch

from models.nets.multi_headed_vae import MultiHeadedVAE


class MultiHeadedVAETrainer:
    def __init__(
        self,
        single_vae_checkpoint: Optional[str] = None,
        multi_vae_checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()
        # Platform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_name = "multi_headed_vae"
        self.run_dir = os.path.join("../output/", self.run_name)
        self.run_time = datetime.now().strftime("%m%d%H%M")

        # Instantiate the model
        self.model = MultiHeadedVAE()

        # Check initialisation conditions
        # if multi_vae_checkpoint is not None:




