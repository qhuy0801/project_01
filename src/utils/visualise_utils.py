import cv2
from matplotlib import pyplot as plt


def visualise_image():
    """

    :return:
    """
    # TODO
    return None


def visualise_image_grayscale():
    """

    :return:
    """
    # TODO
    return None


def visualise_segmentation(_image, _segmentation):
    """
    Visualise segmented image
    :param _image:
    :param _segmentation:
    :return:
    """
    f, ax = plt.subplots(figsize=(8, 6))
    plt.tight_layout()
    ax.imshow(cv2.cvtColor(_image, cv2.COLOR_RGB2GRAY), cmap="gray")
    ax.axis("off")
    segmentation = cv2.resize(_segmentation[0], _image.shape[:-1][::-1], interpolation=cv2.INTER_NEAREST)
    ax.imshow(segmentation, alpha=0.4)
    plt.show()

