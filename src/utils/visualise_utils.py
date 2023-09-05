from matplotlib import pyplot as plt


def plot_chw(chw_tensor):
    """
    Plot single CHW tensor to image
    :param chw_tensor:
    :return:
    """
    plt.imshow(chw_tensor.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.show()


def plot_hwc(hwc_ndarray):
    """
    PLot single HWC numpy array to image
    :param hwc_ndarray:
    :return:
    """
    plt.imshow(hwc_ndarray)
    plt.axis('off')
    plt.show()
