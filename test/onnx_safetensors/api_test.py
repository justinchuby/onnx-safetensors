from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from typing import Any

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnxruntime as ort
import parameterized
import safetensors.numpy
import onnx_ir as ir

import onnx_safetensors
from onnx_safetensors._safetensors_io import _get_shard_filename, _parse_size_string


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


def _create_runnable_test_model() -> onnx.ModelProto:
    """Create a simple runnable ONNX model for testing with ORT."""
    # Create a simple model: output = input + weights
    # Input shape: (2, 3)
    # Weights shape: (2, 3)
    weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Add", ["input", "weights"], ["output"]),
        ],
        "runnable_test",
        inputs=[
            onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 3]),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info(
                "output", onnx.TensorProto.FLOAT, [2, 3]
            ),
        ],
        initializer=[onnx.numpy_helper.from_array(weights, name="weights")],
    )

    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 20)], ir_version=10
    )
    return model


class PublicApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _create_test_model()
        self.model_tensor_dict = _get_model_tensor_dict()
        self.replacement_tensor_dict = _get_replacement_tensor_dict()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tensor_file_path = pathlib.Path(self.temp_dir.name) / "tensor.safetensors"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def assert_ort(
        self,
        model_path: pathlib.Path,
        inputs: dict[str, np.ndarray],
        expected: dict[str, np.ndarray],
    ) -> None:
        """Assert that a model can be run with onnxruntime and produces expected outputs.

        Args:
            model_path: Path to the ONNX model file.
            inputs: Dictionary of input names to numpy arrays.
            expected: Dictionary of output names to expected numpy arrays.
        """
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        outputs = sess.run(None, inputs)
        output_names = [o.name for o in sess.get_outputs()]

        for i, name in enumerate(output_names):
            if name in expected:
                np.testing.assert_allclose(
                    outputs[i], expected[name], rtol=1e-5, atol=1e-5
                )

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

    def test_save_model_creates_onnx_and_safetensors_files(self) -> None:
        model_path = pathlib.Path(self.temp_dir.name) / "model.onnx"
        external_data_path = "weights.safetensors"

        onnx_safetensors.save_model(
            self.model, model_path, external_data=external_data_path
        )

        # Verify both files were created
        self.assertTrue(model_path.exists())
        self.assertTrue(
            (pathlib.Path(self.temp_dir.name) / external_data_path).exists()
        )

        # Load and verify the saved model
        loaded_model = onnx.load(model_path)
        self.assertIsNotNone(loaded_model)

        # Verify safetensors file contains the correct tensors
        tensors = safetensors.numpy.load_file(
            pathlib.Path(self.temp_dir.name) / external_data_path
        )
        for key in tensors:
            np.testing.assert_array_equal(tensors[key], self.model_tensor_dict[key])

    def test_save_model_without_external_data_argument(self) -> None:
        model_path = pathlib.Path(self.temp_dir.name) / "model.onnx"

        # Call save_model without providing external_data
        onnx_safetensors.save_model(self.model, model_path)

        # Verify both files were created with auto-derived names
        self.assertTrue(model_path.exists())
        # Should auto-create model.safetensors
        expected_safetensors = pathlib.Path(self.temp_dir.name) / "model.safetensors"
        self.assertTrue(expected_safetensors.exists())

        # Verify safetensors file contains the correct tensors
        tensors = safetensors.numpy.load_file(expected_safetensors)
        for key in tensors:
            np.testing.assert_array_equal(tensors[key], self.model_tensor_dict[key])

    def test_save_model_can_run_with_ort(self) -> None:
        """Test that saved models can be executed with ONNX Runtime."""
        model = _create_runnable_test_model()
        model_path = pathlib.Path(self.temp_dir.name) / "runnable.onnx"

        onnx_safetensors.save_model(model, model_path)

        # Verify the saved model can run
        input_data = np.array(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32
        )
        weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        expected_output = input_data + weights

        self.assert_ort(
            model_path,
            {"input": input_data},
            {"output": expected_output},
        )

    def test_save_model_without_external_data_with_sharding(self) -> None:
        # Create a model with multiple tensors to test sharding
        tensor1 = np.arange(1000).reshape(100, 10).astype(np.float32)
        tensor2 = np.arange(2000).reshape(200, 10).astype(np.float32)
        tensor3 = np.arange(500).reshape(50, 10).astype(np.float32)

        initializers = [
            onnx.numpy_helper.from_array(tensor1, name="tensor1"),
            onnx.numpy_helper.from_array(tensor2, name="tensor2"),
            onnx.numpy_helper.from_array(tensor3, name="tensor3"),
        ]

        graph = onnx.helper.make_graph(
            [],
            "test_graph",
            inputs=[],
            outputs=[],
            initializer=initializers,
        )
        model = onnx.helper.make_model(graph)

        model_path = pathlib.Path(self.temp_dir.name) / "mymodel.onnx"

        # Call save_model without providing external_data, but with max_shard_size
        onnx_safetensors.save_model(model, model_path, max_shard_size="5KB")

        # Verify model file was created
        self.assertTrue(model_path.exists())

        # Check that shard files were created with auto-derived base name
        shard_files = list(
            pathlib.Path(self.temp_dir.name).glob("mymodel-*.safetensors")
        )
        self.assertGreater(
            len(shard_files), 1, "Expected multiple shard files to be created"
        )

        # Check that index file was created with auto-derived name
        index_file = pathlib.Path(self.temp_dir.name) / "mymodel.safetensors.index.json"
        self.assertTrue(
            index_file.exists(), "Index file should be created when sharding"
        )

        # Verify index file content
        with open(index_file) as f:
            index_data = json.load(f)

        self.assertIn("weight_map", index_data)
        self.assertIn("metadata", index_data)
        self.assertEqual(
            len(index_data["weight_map"]), 3, "Should have 3 tensors in weight map"
        )

        # Verify all tensors are accounted for
        for tensor_name in ["tensor1", "tensor2", "tensor3"]:
            self.assertIn(tensor_name, index_data["weight_map"])

    def test_save_model_with_sharding_can_run_with_ort(self) -> None:
        """Test that sharded models can be executed with ONNX Runtime."""
        # Create a runnable model with larger weights to force sharding
        # Use compatible shapes: (1, 2000) and (1, 4000)
        weights1 = np.arange(2000).astype(np.float32).reshape(1, 2000)
        weights2 = np.arange(4000).astype(np.float32).reshape(1, 4000)

        # Create a model: add weights1, concat to expand, then add weights2
        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node("Add", ["input", "weights1"], ["temp"]),
                onnx.helper.make_node("Concat", ["temp", "temp"], ["expanded"], axis=1),
                onnx.helper.make_node("Add", ["expanded", "weights2"], ["output"]),
            ],
            "sharded_runnable",
            inputs=[
                onnx.helper.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, [1, 2000]
                ),
            ],
            outputs=[
                onnx.helper.make_tensor_value_info(
                    "output", onnx.TensorProto.FLOAT, [1, 4000]
                ),
            ],
            initializer=[
                onnx.numpy_helper.from_array(weights1, name="weights1"),
                onnx.numpy_helper.from_array(weights2, name="weights2"),
            ],
        )
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 20)], ir_version=10
        )

        model_path = pathlib.Path(self.temp_dir.name) / "sharded_runnable.onnx"

        # Save with small shard size to force sharding
        onnx_safetensors.save_model(model, model_path, max_shard_size="10KB")

        # Verify sharding occurred
        shard_files = list(
            pathlib.Path(self.temp_dir.name).glob("sharded_runnable-*.safetensors")
        )
        self.assertGreater(len(shard_files), 1, "Expected sharding to occur")

        # Verify the saved model can still run
        input_data = np.ones((1, 2000), dtype=np.float32)
        temp_result = input_data + weights1
        expanded = np.concatenate([temp_result, temp_result], axis=1)
        expected_output = expanded + weights2

        self.assert_ort(
            model_path,
            {"input": input_data},
            {"output": expected_output},
        )

    def test_save_model_requires_safetensors_extension(self) -> None:
        model_path = pathlib.Path(self.temp_dir.name) / "model.onnx"
        invalid_external_data_path = "weights.bin"

        with self.assertRaises(ValueError) as context:
            onnx_safetensors.save_model(
                self.model, model_path, external_data=invalid_external_data_path
            )

        self.assertIn(".safetensors", str(context.exception))

    def test_save_model_with_size_threshold(self) -> None:
        model_path = pathlib.Path(self.temp_dir.name) / "model.onnx"
        external_data_path = "weights.safetensors"

        # Use a high threshold to exclude all tensors
        onnx_safetensors.save_model(
            self.model,
            model_path,
            external_data=external_data_path,
            size_threshold=1000,
        )

        # Verify model file was created
        self.assertTrue(model_path.exists())

        # Verify safetensors file exists but is empty or minimal
        safetensors_path = pathlib.Path(self.temp_dir.name) / external_data_path
        self.assertFalse(safetensors_path.exists())

    def test_save_model_external_data_is_relative_path(self) -> None:
        model_path = pathlib.Path(self.temp_dir.name) / "model.onnx"
        external_data_path = "weights.safetensors"

        onnx_safetensors.save_model(
            self.model, model_path, external_data=external_data_path
        )

        # Load the model and check that external data references are relative
        loaded_model = onnx.load(model_path)
        for initializer in loaded_model.graph.initializer:
            if not initializer.HasField("data_location"):
                continue
            if initializer.data_location != onnx.TensorProto.EXTERNAL:
                continue
            for entry in initializer.external_data:
                if entry.key == "location":
                    # The location should be just the filename, not an absolute path
                    self.assertEqual(entry.value, external_data_path)


def _create_test_ir_model(dtype: ir.DataType) -> ir.Model:
    input_ = ir.val(
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

    def assert_ort(
        self,
        model_path: pathlib.Path,
        inputs: dict[str, np.ndarray],
        expected: dict[str, np.ndarray],
    ) -> None:
        """Assert that a model can be run with onnxruntime and produces expected outputs.

        Args:
            model_path: Path to the ONNX model file.
            inputs: Dictionary of input names to numpy arrays.
            expected: Dictionary of output names to expected numpy arrays.
        """
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        outputs = sess.run(None, inputs)
        output_names = [o.name for o in sess.get_outputs()]

        for i, name in enumerate(output_names):
            if name in expected:
                np.testing.assert_allclose(
                    outputs[i], expected[name], rtol=1e-5, atol=1e-5
                )

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
            (ir.DataType.FLOAT8E4M3FNUZ.name, ir.DataType.FLOAT8E4M3FNUZ),
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

    def test_save_model_from_ir_model(self) -> None:
        model = _create_test_ir_model(ir.DataType.FLOAT)
        model_path = pathlib.Path(self.temp_dir.name) / "model.onnx"
        external_data_path = "weights.safetensors"

        onnx_safetensors.save_model(model, model_path, external_data=external_data_path)

        # Verify both files were created
        self.assertTrue(model_path.exists())
        self.assertTrue(
            (pathlib.Path(self.temp_dir.name) / external_data_path).exists()
        )

        # Load and verify the saved model
        loaded_model = onnx.load(model_path)
        self.assertIsNotNone(loaded_model)

        # Verify safetensors file contains the correct tensor
        with open(pathlib.Path(self.temp_dir.name) / external_data_path, "rb") as f:
            tensors = safetensors.deserialize(f.read())
        tensor = ir.tensor([0, 1, 6], dtype=ir.DataType.FLOAT)
        self.assertEqual(tensors[0][1]["data"], tensor.tobytes())

    def test_save_model_from_ir_model_requires_safetensors_extension(self) -> None:
        model = _create_test_ir_model(ir.DataType.FLOAT)
        model_path = pathlib.Path(self.temp_dir.name) / "model.onnx"
        invalid_external_data_path = "weights.bin"

        with self.assertRaises(ValueError) as context:
            onnx_safetensors.save_model(
                model, model_path, external_data=invalid_external_data_path
            )

        self.assertIn(".safetensors", str(context.exception))

    def test_save_model_from_ir_model_with_size_threshold(self) -> None:
        model = _create_test_ir_model(ir.DataType.FLOAT)
        model_path = pathlib.Path(self.temp_dir.name) / "model.onnx"
        external_data_path = "weights.safetensors"

        # Use a high threshold to exclude all tensors
        onnx_safetensors.save_model(
            model, model_path, external_data=external_data_path, size_threshold=1000
        )

        # Verify model file was created
        self.assertTrue(model_path.exists())

        # Verify safetensors file exists
        safetensors_path = pathlib.Path(self.temp_dir.name) / external_data_path
        self.assertFalse(safetensors_path.exists())

    def test_save_file_with_max_shard_size(self) -> None:
        # Create a model with multiple tensors to test sharding
        tensor1 = np.arange(1000).reshape(100, 10).astype(np.float32)
        tensor2 = np.arange(2000).reshape(200, 10).astype(np.float32)
        tensor3 = np.arange(500).reshape(50, 10).astype(np.float32)

        initializers = [
            onnx.numpy_helper.from_array(tensor1, name="tensor1"),
            onnx.numpy_helper.from_array(tensor2, name="tensor2"),
            onnx.numpy_helper.from_array(tensor3, name="tensor3"),
        ]

        graph = onnx.helper.make_graph(
            [],
            "test_graph",
            inputs=[],
            outputs=[],
            initializer=initializers,
        )
        model = onnx.helper.make_model(graph)

        # Save with small shard size to force sharding
        tensor_file_path = pathlib.Path(self.temp_dir.name) / "weights.safetensors"
        # Each tensor is ~4KB, so setting max_shard_size to 5KB should create multiple shards
        onnx_safetensors.save_file(model, tensor_file_path, max_shard_size="5KB")

        # Check that multiple shard files were created
        shard_files = list(
            pathlib.Path(self.temp_dir.name).glob("weights-*.safetensors")
        )
        self.assertGreater(
            len(shard_files), 1, "Expected multiple shard files to be created"
        )

        # Check that index file was created
        index_file = pathlib.Path(self.temp_dir.name) / "weights.safetensors.index.json"
        self.assertTrue(
            index_file.exists(), "Index file should be created when sharding"
        )

        # Verify index file content
        with open(index_file) as f:
            index_data = json.load(f)

        self.assertIn("weight_map", index_data)
        self.assertIn("metadata", index_data)
        self.assertEqual(
            len(index_data["weight_map"]), 3, "Should have 3 tensors in weight map"
        )

        # Verify all tensors are accounted for
        for tensor_name in ["tensor1", "tensor2", "tensor3"]:
            self.assertIn(tensor_name, index_data["weight_map"])

    def test_save_file_with_max_shard_size_no_sharding_needed(self) -> None:
        # Create a small model that doesn't need sharding
        model = _create_test_model()
        tensor_file_path = pathlib.Path(self.temp_dir.name) / "weights.safetensors"

        # Save with large shard size - no sharding should occur
        onnx_safetensors.save_file(model, tensor_file_path, max_shard_size="100GB")

        # Check that only one file was created (no shard suffix)
        self.assertTrue(tensor_file_path.exists())

        # Check that no index file was created
        index_file = pathlib.Path(self.temp_dir.name) / "weights.safetensors.index.json"
        self.assertFalse(index_file.exists())

        # Check that no shard files were created
        shard_files = list(
            pathlib.Path(self.temp_dir.name).glob("weights-*.safetensors")
        )
        self.assertEqual(len(shard_files), 0)

    def test_save_model_with_max_shard_size(self) -> None:
        # Create a model with multiple tensors
        tensor1 = np.arange(1000).reshape(100, 10).astype(np.float32)
        tensor2 = np.arange(2000).reshape(200, 10).astype(np.float32)

        initializers = [
            onnx.numpy_helper.from_array(tensor1, name="tensor1"),
            onnx.numpy_helper.from_array(tensor2, name="tensor2"),
        ]

        graph = onnx.helper.make_graph(
            [],
            "test_graph",
            inputs=[],
            outputs=[],
            initializer=initializers,
        )
        model = onnx.helper.make_model(graph)

        model_path = pathlib.Path(self.temp_dir.name) / "model.onnx"
        external_data_path = "weights.safetensors"

        # Save with small shard size
        onnx_safetensors.save_model(
            model, model_path, external_data=external_data_path, max_shard_size="5KB"
        )

        # Verify model file was created
        self.assertTrue(model_path.exists())

        # Check that shard files were created
        shard_files = list(
            pathlib.Path(self.temp_dir.name).glob("weights-*.safetensors")
        )
        self.assertGreater(len(shard_files), 0)

        # Check that index file was created
        index_file = pathlib.Path(self.temp_dir.name) / "weights.safetensors.index.json"
        self.assertTrue(index_file.exists())

    def test_save_model_with_sharding_from_ir_can_run_with_ort(self) -> None:
        """Test that sharded IR models can be executed with ONNX Runtime."""
        # Create a runnable IR model with larger weights to force sharding
        # Using shape (1, 2000) to force sharding while keeping compatible shapes
        weights1 = np.arange(2000).astype(np.float32).reshape(1, 2000)
        weights2 = np.arange(4000).astype(np.float32).reshape(1, 4000)

        # Build graph using IR
        input_val = ir.val(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([1, 2000]),
        )
        weights1_val = ir.val(
            name="weights1",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([1, 2000]),
        )
        weights1_val.const_value = ir.tensor(weights1, name="weights1")
        weights2_val = ir.val(
            name="weights2",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([1, 4000]),
        )
        weights2_val.const_value = ir.tensor(weights2, name="weights2")

        add1 = ir.Node("", "Add", [input_val, weights1_val])
        # Concatenate the result to make it (1, 4000) for the second add
        concat = ir.Node(
            "",
            "Concat",
            [add1.outputs[0], add1.outputs[0]],
            attributes={"axis": ir.AttrInt64("axis", 1)},
        )
        add2 = ir.Node("", "Add", [concat.outputs[0], weights2_val])

        model = ir.Model(
            ir.Graph(
                (input_val,),
                add2.outputs,
                nodes=(add1, concat, add2),
                initializers=(weights1_val, weights2_val),
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        model_path = pathlib.Path(self.temp_dir.name) / "ir_sharded.onnx"

        # Save with small shard size to force sharding
        onnx_safetensors.save_model(model, model_path, max_shard_size="10KB")

        # Verify sharding occurred
        shard_files = list(
            pathlib.Path(self.temp_dir.name).glob("ir_sharded-*.safetensors")
        )
        self.assertGreater(len(shard_files), 1, "Expected sharding to occur")

        # Verify the saved model can still run
        input_data = np.ones((1, 2000), dtype=np.float32)
        temp_result = input_data + weights1
        concat_result = np.concatenate([temp_result, temp_result], axis=1)
        expected_output = concat_result + weights2

        self.assert_ort(
            model_path,
            {"input": input_data},
            {"output": expected_output},
        )

    def test_parse_size_string(self) -> None:
        # Test the size string parsing
        self.assertEqual(_parse_size_string("5GB"), 5 * 1000**3)
        self.assertEqual(_parse_size_string("100MB"), 100 * 1000**2)
        self.assertEqual(_parse_size_string("1KB"), 1000)
        self.assertEqual(_parse_size_string("512B"), 512)
        self.assertEqual(_parse_size_string(1000), 1000)

        # Test invalid formats
        with self.assertRaises(ValueError):
            _parse_size_string("invalid")
        with self.assertRaises(ValueError):
            _parse_size_string("5XB")

    def test_get_shard_filename(self) -> None:
        # Test shard filename generation
        self.assertEqual(
            _get_shard_filename("model.safetensors", 1, 3),
            "model-00001-of-00003.safetensors",
        )
        self.assertEqual(
            _get_shard_filename("model.safetensors", 10, 100),
            "model-00010-of-00100.safetensors",
        )
        self.assertEqual(
            _get_shard_filename("model.safetensors", 1, 1),
            "model.safetensors",
        )


if __name__ == "__main__":
    unittest.main()
