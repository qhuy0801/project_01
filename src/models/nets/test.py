from torchinfo import summary

from models.nets.vae_v2 import VAE_v2
from models.nets.vae_v3 import VAE_v3
from models.nets.vae_v4 import Multi_headed_VAE_v1

if __name__ == '__main__':
    model = VAE_v2()
    print(summary(model, (1, 3, 128, 128)))
