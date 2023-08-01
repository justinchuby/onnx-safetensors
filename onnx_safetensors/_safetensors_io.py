from __future__ import annotations

import itertools
import os
import sys
from typing import Callable, Iterable

import onnx
import safetensors
import safetensors.numpy

from onnx_safetensors import _external_data_helper


def load_safetensors_file(
    model_proto: onnx.ModelProto, tensor_file: str | os.PathLike
) -> None:
    with safetensors.safe_open(tensor_file, "numpy") as f:
        keys = f.keys()
        for tensor in _external_data_helper.get_all_tensors(model_proto):
            name = tensor.name
            if name not in keys:
                continue
            place_holder = onnx.helper.make_tensor(
                name,
                tensor.data_type,
                tensor.dims,
                vals=f.get_tensor(name),
            )
            tensor.raw_data = place_holder.raw_data
            _external_data_helper.set_external_data_flag(tensor, False)


def load_safetensors(model_proto: onnx.ModelProto, data: bytes) -> None:
    tensor_dict = safetensors.numpy.load(data)

    for tensor in _external_data_helper.get_all_tensors(model_proto):
        name = tensor.name
        if (external_tensor := tensor_dict.get(name)) is None:
            continue
        place_holder = onnx.helper.make_tensor(
            name,
            tensor.data_type,
            tensor.dims,
            vals=external_tensor,
        )
        tensor.raw_data = place_holder.raw_data
        _external_data_helper.set_external_data_flag(tensor, False)


def save_safetensors_file(
    model_proto: onnx.ModelProto,
    tensor_file: str | os.PathLike,
    size_threshold: int = 1024,
    convert_attributes: bool = False,
) -> None:
    if convert_attributes:
        tensors = _external_data_helper.get_all_tensors(model_proto)
    else:
        tensors = _external_data_helper.get_initializer_tensors(model_proto)

    tensor_dict = {}

    for tensor in tensors:
        name = tensor.name
        if not (
            tensor.HasField("raw_data")
            and sys.getsizeof(tensor.raw_data) >= size_threshold
        ):
            continue
        tensor_dict[name] = onnx.numpy_helper.to_array(tensor)
        _external_data_helper.set_external_data_flag(tensor, True)

    safetensors.numpy.save_file(tensor_dict, tensor_file)


# TODO: Allow saving the model file and or returning the bytes and model proto
