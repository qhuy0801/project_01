import onnx
import onnxruntime
import torch


def load_pt(_pt_path, _device):
    return torch.load(_pt_path, map_location=_device)


def load_onnx(_onnx_path):
    model = onnx.load(_onnx_path)
    onnx.checker.check_model(model)
    return model


def init_session_onnx(_model):
    return onnxruntime.InferenceSession(_model.SerializeToString(), None)
