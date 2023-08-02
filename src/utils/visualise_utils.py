import cv2
from matplotlib import pyplot as plt


def visualise_layers(*args):
    """
    Visualise layers based on tuples configurations
    With layer in the shape of (H, W, 3) or (H, W, 4), cmap will be ignored
    :param args: List of tuples, each represent a layer (image_matrix, cmap, opacity)
    :return:
    """
    f, ax = plt.subplots(figsize=(8, 6))
    plt.tight_layout()
    ax.axis("off")
    for arg in args:
        ax.imshow(arg[0], cmap=arg[1], alpha=arg[2])
    plt.show()
