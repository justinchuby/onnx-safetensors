from __future__ import annotations

import pathlib
import tempfile
import unittest
from typing import Any

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import parameterized
import safetensors.numpy

from onnx_safetensors import _safetensors_io


def _create_tensor(value: Any, tensor_name: str) -> onnx.TensorProto:
    tensor = onnx.numpy_helper.from_array(np.array(value))
    tensor.name = tensor_name
    return tensor


def _create_test_graph() -> onnx.GraphProto:
    tensor_dict = _get_model_tensor_dict()
    initializer_value = tensor_dict["input_value"]
    attribute_value = tensor_dict["attribute_value"]
    constant_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["values"],
        value=_create_tensor(attribute_value, "attribute_value"),
    )

    initializers = [_create_tensor(initializer_value, "input_value")]
    inputs = [
        onnx.helper.make_tensor_value_info(
            "input_value", onnx.TensorProto.FLOAT, initializer_value.shape
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
        "input_value": np.arange(6).reshape(3, 2).astype(np.float32) + 42,
        "attribute_value": np.arange(6).reshape(2, 3).astype(np.float32) + 24,
        "unused_value": np.array([1.0, 2.0, 3.0]),
    }


def _get_model_tensor_dict() -> dict[str, np.ndarray]:
    initializer_value = np.arange(6).reshape(3, 2).astype(np.float32) + 512
    attribute_value = np.arange(6).reshape(2, 3).astype(np.float32) + 256
    return {
        "attribute_value": attribute_value,
        "input_value": initializer_value,
    }


class SafeTensorsIoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _create_test_model()
        self.graph = _create_test_graph()
        self.model_tensor_dict = _get_model_tensor_dict()
        self.replacement_tensor_dict = _get_replacement_tensor_dict()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tensor_file_path = pathlib.Path(self.temp_dir.name) / "tensor.safetensors"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_file_to_model(self) -> None:
        safetensors.numpy.save_file(self.replacement_tensor_dict, self.tensor_file_path)
        actual = _safetensors_io.load_file(self.model, self.tensor_file_path)
        expected = {"attribute_value", "input_value"}
        self.assertEqual(actual, expected)

    def test_load_file_to_graph(self) -> None:
        safetensors.numpy.save_file(self.replacement_tensor_dict, self.tensor_file_path)
        actual = _safetensors_io.load_file(self.graph, self.tensor_file_path)
        expected = {"attribute_value", "input_value"}
        self.assertEqual(actual, expected)

    def test_load_to_model(self) -> None:
        tensors = safetensors.numpy.save(self.replacement_tensor_dict)
        actual = _safetensors_io.load(self.model, tensors)
        expected = {"attribute_value", "input_value"}
        self.assertEqual(actual, expected)

    def test_load_to_graph(self) -> None:
        tensors = safetensors.numpy.save(self.replacement_tensor_dict)
        actual = _safetensors_io.load(self.graph, tensors)
        expected = {"attribute_value", "input_value"}
        self.assertEqual(actual, expected)

    @parameterized.parameterized.expand(
        [
            (True, {"attribute_value", "input_value"}),
            (False, {"input_value"}),
        ]
    )
    def test_save_file_from_model(self, convert_attributes, expected) -> None:
        actual = _safetensors_io.save_file(
            self.model, self.tensor_file_path, convert_attributes=convert_attributes
        )
        self.assertEqual(actual, expected)
        tensors = safetensors.numpy.load_file(self.tensor_file_path)
        self.assertEqual(set(tensors.keys()), expected)
        for key in expected:
            np.testing.assert_array_equal(tensors[key], self.model_tensor_dict[key])

    @parameterized.parameterized.expand(
        [
            (True, {"attribute_value", "input_value"}),
            (False, {"input_value"}),
        ]
    )
    def test_save_file_from_graph(self, convert_attributes, expected) -> None:
        actual = _safetensors_io.save_file(
            self.graph, self.tensor_file_path, convert_attributes=convert_attributes
        )
        self.assertEqual(actual, expected)
        tensors = safetensors.numpy.load_file(self.tensor_file_path)
        self.assertEqual(set(tensors.keys()), expected)
        for key in expected:
            np.testing.assert_array_equal(tensors[key], self.model_tensor_dict[key])

    @unittest.skip("Not implemented")
    def test_save_file_clears_raw_data_when_strip_data_is_true(self):
        raise NotImplementedError()

    def test_strip_raw_data_clears_specified_raw_data(self) -> None:
        _safetensors_io.strip_raw_data(self.model, {"input_value"})
        # TODO: Test that raw data is cleared


# TODO: Test all ONNX data types
