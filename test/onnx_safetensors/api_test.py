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
import onnx_ir as ir

import onnx_safetensors


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


class PublicApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _create_test_model()
        self.model_tensor_dict = _get_model_tensor_dict()
        self.replacement_tensor_dict = _get_replacement_tensor_dict()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tensor_file_path = pathlib.Path(self.temp_dir.name) / "tensor.safetensors"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_file_to_model(self) -> None:
        safetensors.numpy.save_file(self.replacement_tensor_dict, self.tensor_file_path)
        proto = onnx_safetensors.load_file(self.model, self.tensor_file_path)
        model = ir.serde.deserialize_model(proto)

        np.testing.assert_equal(
            model.graph.initializers["initializer_value"].const_value,
            self.replacement_tensor_dict["initializer_value"],
        )

    def test_load_to_model(self) -> None:
        tensors = safetensors.numpy.save(self.replacement_tensor_dict)
        proto = onnx_safetensors.load(self.model, tensors)
        model = ir.serde.deserialize_model(proto)

        np.testing.assert_equal(
            model.graph.initializers["initializer_value"].const_value,
            self.replacement_tensor_dict["initializer_value"],
        )

    def test_save_file_from_model(self) -> None:
        _ = onnx_safetensors.save_file(self.model, self.tensor_file_path)
        tensors = safetensors.numpy.load_file(self.tensor_file_path)
        for key in tensors:
            np.testing.assert_array_equal(tensors[key], self.model_tensor_dict[key])


def _create_test_ir_model(dtype: ir.DataType) -> ir.Model:
    input_ = ir.Input(
        name="initializer_value", type=ir.TensorType(dtype), shape=ir.Shape((3,))
    )
    input_.const_value = ir.tensor([0, 1, 6], dtype=dtype, name="initializer_value")

    identity = ir.Node("", "Identity", [input_])
    model = ir.Model(
        ir.Graph(
            (input_,),
            identity.outputs,
            nodes=(identity,),
            initializers=(input_,),
            opset_imports={"": 20},
        ),
        ir_version=10,
    )

    return model


class PublicIrApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model_tensor_dict = _get_model_tensor_dict()
        self.replacement_tensor_dict = _get_replacement_tensor_dict()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tensor_file_path = pathlib.Path(self.temp_dir.name) / "tensor.safetensors"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_file_to_ir_model(self) -> None:
        safetensors.numpy.save_file(self.replacement_tensor_dict, self.tensor_file_path)
        proto = _create_test_model()
        model = ir.from_proto(proto)
        model = onnx_safetensors.load_file(model, self.tensor_file_path)

        np.testing.assert_equal(
            model.graph.initializers["initializer_value"].const_value,
            self.replacement_tensor_dict["initializer_value"],
        )

    def test_load_to_ir_model(self) -> None:
        tensors = safetensors.numpy.save(self.replacement_tensor_dict)
        proto = _create_test_model()
        model = ir.from_proto(proto)
        model = onnx_safetensors.load(model, tensors)

        np.testing.assert_equal(
            model.graph.initializers["initializer_value"].const_value,
            self.replacement_tensor_dict["initializer_value"],
        )

    @parameterized.parameterized.expand(
        [
            (ir.DataType.FLOAT.name, ir.DataType.FLOAT),
            (ir.DataType.UINT8.name, ir.DataType.UINT8),
            (ir.DataType.INT8.name, ir.DataType.INT8),
            (ir.DataType.UINT16.name, ir.DataType.UINT16),
            (ir.DataType.INT16.name, ir.DataType.INT16),
            (ir.DataType.INT32.name, ir.DataType.INT32),
            (ir.DataType.INT64.name, ir.DataType.INT64),
            (ir.DataType.BOOL.name, ir.DataType.BOOL),
            (ir.DataType.FLOAT16.name, ir.DataType.FLOAT16),
            (ir.DataType.DOUBLE.name, ir.DataType.DOUBLE),
            (ir.DataType.UINT32.name, ir.DataType.UINT32),
            (ir.DataType.UINT64.name, ir.DataType.UINT64),
            # (ir.DataType.COMPLEX64.name, ir.DataType.COMPLEX64) ,
            # (ir.DataType.COMPLEX128.name, ir.DataType.COMPLEX128) ,
            (ir.DataType.BFLOAT16.name, ir.DataType.BFLOAT16),
            (ir.DataType.FLOAT8E4M3FN.name, ir.DataType.FLOAT8E4M3FN),
            (ir.DataType.FLOAT8E4M3FNUZ.name, ir.DataType.FLOAT8E4M3FNUZ),
            (ir.DataType.FLOAT8E5M2.name, ir.DataType.FLOAT8E5M2),
            (ir.DataType.FLOAT8E5M2FNUZ.name, ir.DataType.FLOAT8E5M2FNUZ),
            (ir.DataType.UINT4.name, ir.DataType.UINT4),
            (ir.DataType.INT4.name, ir.DataType.INT4),
            (ir.DataType.FLOAT4E2M1.name, ir.DataType.FLOAT4E2M1),
        ]
    )
    def test_save_from_ir_model(self, _: str, dtype: ir.DataType) -> None:
        model = _create_test_ir_model(dtype)
        data = onnx_safetensors.save(model)
        tensors = safetensors.deserialize(data)
        tensor = ir.tensor([0, 1, 6], dtype=dtype)
        self.assertEqual(tensors[0][1]["data"], tensor.tobytes())

    @parameterized.parameterized.expand(
        [
            (ir.DataType.FLOAT.name, ir.DataType.FLOAT),
            (ir.DataType.UINT8.name, ir.DataType.UINT8),
            (ir.DataType.INT8.name, ir.DataType.INT8),
            (ir.DataType.UINT16.name, ir.DataType.UINT16),
            (ir.DataType.INT16.name, ir.DataType.INT16),
            (ir.DataType.INT32.name, ir.DataType.INT32),
            (ir.DataType.INT64.name, ir.DataType.INT64),
            (ir.DataType.BOOL.name, ir.DataType.BOOL),
            (ir.DataType.FLOAT16.name, ir.DataType.FLOAT16),
            (ir.DataType.DOUBLE.name, ir.DataType.DOUBLE),
            (ir.DataType.UINT32.name, ir.DataType.UINT32),
            (ir.DataType.UINT64.name, ir.DataType.UINT64),
            # (ir.DataType.COMPLEX64.name, ir.DataType.COMPLEX64) ,
            # (ir.DataType.COMPLEX128.name, ir.DataType.COMPLEX128) ,
            (ir.DataType.BFLOAT16.name, ir.DataType.BFLOAT16),
            (ir.DataType.FLOAT8E4M3FN.name, ir.DataType.FLOAT8E4M3FN),
            # (ir.DataType.FLOAT8E4M3FNUZ.name, ir.DataType.FLOAT8E4M3FNUZ),
            # TODO: FLOAT8E4M3FNUZ support in ONNX IR was fixed in 0.3. Enable when it is released
            (ir.DataType.FLOAT8E5M2.name, ir.DataType.FLOAT8E5M2),
            (ir.DataType.FLOAT8E5M2FNUZ.name, ir.DataType.FLOAT8E5M2FNUZ),
            (ir.DataType.UINT4.name, ir.DataType.UINT4),
            (ir.DataType.INT4.name, ir.DataType.INT4),
            (ir.DataType.FLOAT4E2M1.name, ir.DataType.FLOAT4E2M1),
        ]
    )
    def test_save_file_from_ir_model(self, _: str, dtype: ir.DataType) -> None:
        model = _create_test_ir_model(dtype)
        _ = onnx_safetensors.save_file(model, self.tensor_file_path)
        with open(self.tensor_file_path, "rb") as f:
            tensors = safetensors.deserialize(f.read())
        tensor = ir.tensor([0, 1, 6], dtype=dtype)
        self.assertEqual(tensors[0][1]["data"], tensor.tobytes())


if __name__ == "__main__":
    unittest.main()
