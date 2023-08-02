import onnx
import onnxruntime
import torch


def load_pt(_pt_path, _device):
    """
    Load model which is pytorch
    :param _pt_path:
    :param _device:
    :return:
    """
    return torch.load(_pt_path, map_location=_device)
