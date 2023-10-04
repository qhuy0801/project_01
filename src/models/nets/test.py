from diffusers import AutoencoderKL
from torchinfo import summary

from models.nets.vae_v3 import VAE_v3

if __name__ == '__main__':
    print(summary(VAE_v3(), (1, 3, 128, 128)))
