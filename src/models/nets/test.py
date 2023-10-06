from torchinfo import summary

from models.nets.vae_v4 import Multi_headed_VAE_v1

if __name__ == '__main__':
    print(summary(Multi_headed_VAE_v1(input_size= 256), (1, 3, 256, 256)))
