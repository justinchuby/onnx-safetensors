from __future__ import annotations

import pathlib
import tempfile
import unittest
from typing import Any

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import safetensors.numpy
from onnxscript import ir

from onnx_safetensors import _safetensors_io


def _create_tensor(value: Any, tensor_name: str) -> onnx.TensorProto:
    tensor = onnx.numpy_helper.from_array(np.array(value))
    tensor.name = tensor_name
    return tensor


def _create_test_graph() -> onnx.GraphProto:
    tensor_dict = _get_model_tensor_dict()
    initializer_value = tensor_dict["initializer_value"]
    attribute_value = tensor_dict["attribute_value"]
    constant_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["values"],
        value=_create_tensor(attribute_value, "attribute_value"),
    )

    initializers = [_create_tensor(initializer_value, "initializer_value")]
    inputs = [
        onnx.helper.make_tensor_value_info(
            "initializer_value", onnx.TensorProto.FLOAT, initializer_value.shape
        )
    ]

    graph = onnx.helper.make_graph(
        [constant_node],
        "test_graph",
        inputs=inputs,
        outputs=[],
        initializer=initializers,
    )
    return graph


def _create_test_model() -> onnx.ModelProto:
    model = onnx.helper.make_model(_create_test_graph())

    return model


def _get_replacement_tensor_dict() -> dict[str, np.ndarray]:
    return {
        "initializer_value": np.arange(6).reshape(3, 2).astype(np.float32) + 42,
        "attribute_value": np.arange(6).reshape(2, 3).astype(np.float32) + 24,
        "unused_value": np.array([1.0, 2.0, 3.0]),
    }


def _get_model_tensor_dict() -> dict[str, np.ndarray]:
    initializer_value = np.arange(6).reshape(3, 2).astype(np.float32) + 512
    attribute_value = np.arange(6).reshape(2, 3).astype(np.float32) + 256
    return {
        "attribute_value": attribute_value,
        "initializer_value": initializer_value,
    }


class SafeTensorsIoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _create_test_model()
        self.model_ir = ir.serde.deserialize_model(self.model)
        self.model_tensor_dict = _get_model_tensor_dict()
        self.replacement_tensor_dict = _get_replacement_tensor_dict()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tensor_file_path = pathlib.Path(self.temp_dir.name) / "tensor.safetensors"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_file_to_model(self) -> None:
        safetensors.numpy.save_file(self.replacement_tensor_dict, self.tensor_file_path)
        proto = _safetensors_io.load_file(self.model, self.tensor_file_path)
        model = ir.serde.deserialize_model(proto)

        np.testing.assert_equal(
            model.graph.initializers["initializer_value"].const_value,
            self.replacement_tensor_dict["initializer_value"],
        )

    def test_load_to_model(self) -> None:
        tensors = safetensors.numpy.save(self.replacement_tensor_dict)
        proto = _safetensors_io.load(self.model, tensors)
        model = ir.serde.deserialize_model(proto)

        np.testing.assert_equal(
            model.graph.initializers["initializer_value"].const_value,
            self.replacement_tensor_dict["initializer_value"],
        )

    def test_save_file_from_model(self) -> None:
        _ = _safetensors_io.save_file(self.model, self.tensor_file_path)
        tensors = safetensors.numpy.load_file(self.tensor_file_path)
        for key in tensors:
            np.testing.assert_array_equal(tensors[key], self.model_tensor_dict[key])


# TODO: Test all ONNX data types


if __name__ == "__main__":
    unittest.main()
