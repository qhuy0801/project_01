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


def load_onnx(_onnx_path):
    """
    Load an ONNX model
    :param _onnx_path:
    :return:
    """
    model = onnx.load(_onnx_path)
    onnx.checker.check_model(model)
    return model


def init_session_onnx(_model):
    """
    Initialise an inference session using ONNX model
    :param _model:
    :return:
    """
    return onnxruntime.InferenceSession(_model.SerializeToString(), None)
