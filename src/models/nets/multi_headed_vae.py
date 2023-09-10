from copy import deepcopy

from models.nets.vae import VAE


class MultHeadedVAE(VAE):
    def __init__(self, input_size: int = None, dims: [int] = None, latent_dim: int = None, *args, **kwargs) -> None:
        super().__init__(input_size, dims, latent_dim, *args, **kwargs)
        # Make 2 more copy of the decoder
        self.decoder_1 = deepcopy(self.decoder)
        self.decoder_2 = deepcopy(self.decoder)
