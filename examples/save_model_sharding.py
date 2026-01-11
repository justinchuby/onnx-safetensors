"""Example demonstrating save_model and model sharding functionality.

This example shows how to:
1. Save an ONNX model with safetensors weights using save_model
2. Shard large models across multiple safetensors files
3. Load and verify sharded models with ONNX Runtime
"""

import glob
import json
import os

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnxruntime as ort

import onnx_safetensors


def create_example_model(large: bool = False) -> onnx.ModelProto:
    """Create an example ONNX model for demonstration.

    Args:
        large: If True, creates a larger model to demonstrate sharding.

    Returns:
        An ONNX model.
    """
    if large:
        # Create a larger model with multiple weight matrices to demonstrate sharding
        weights1 = np.random.randn(1000, 1000).astype(np.float32)  # ~4MB
        weights2 = np.random.randn(1000, 2000).astype(np.float32)  # ~8MB
        weights3 = np.random.randn(2000, 1000).astype(np.float32)  # ~8MB

        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node("MatMul", ["input", "weights1"], ["temp1"]),
                onnx.helper.make_node("MatMul", ["temp1", "weights2"], ["temp2"]),
                onnx.helper.make_node("MatMul", ["temp2", "weights3"], ["output"]),
            ],
            "large_model",
            inputs=[
                onnx.helper.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, [1, 1000]
                ),
            ],
            outputs=[
                onnx.helper.make_tensor_value_info(
                    "output", onnx.TensorProto.FLOAT, [1, 1000]
                ),
            ],
            initializer=[
                onnx.numpy_helper.from_array(weights1, name="weights1"),
                onnx.numpy_helper.from_array(weights2, name="weights2"),
                onnx.numpy_helper.from_array(weights3, name="weights3"),
            ],
        )
    else:
        # Create a simple model
        weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node("Add", ["input", "weights"], ["output"]),
            ],
            "simple_model",
            inputs=[
                onnx.helper.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, [2, 3]
                ),
            ],
            outputs=[
                onnx.helper.make_tensor_value_info(
                    "output", onnx.TensorProto.FLOAT, [2, 3]
                ),
            ],
            initializer=[onnx.numpy_helper.from_array(weights, name="weights")],
        )

    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 14)], ir_version=10
    )
    return model


def example_basic_save_model():
    """Example 1: Basic usage of save_model."""
    print("Example 1: Basic save_model usage")
    print("=" * 50)

    # Create a simple model
    model = create_example_model(large=False)

    # Save model and weights
    # This creates:
    # - simple_model.onnx (ONNX model file)
    # - simple_model.safetensors (weights file)
    onnx_safetensors.save_model(model, "simple_model.onnx")

    print("✓ Saved simple_model.onnx and simple_model.safetensors")

    # Load and verify the model with ONNX Runtime
    sess = ort.InferenceSession("simple_model.onnx", providers=["CPUExecutionProvider"])
    input_data = np.ones((2, 3), dtype=np.float32)
    outputs = sess.run(None, {"input": input_data})

    print("✓ Model runs successfully with ONNX Runtime")
    print(f"  Output shape: {outputs[0].shape}")
    print()


def example_custom_weights_file():
    """Example 2: Specify a custom name for the weights file."""
    print("Example 2: Custom weights file name")
    print("=" * 50)

    model = create_example_model(large=False)

    # Save with custom weights file name
    # This creates:
    # - my_model.onnx
    # - custom_weights.safetensors
    onnx_safetensors.save_model(
        model, "my_model.onnx", external_data="custom_weights.safetensors"
    )

    print("✓ Saved my_model.onnx with custom_weights.safetensors")
    print()


def example_model_sharding():
    """Example 3: Shard a large model across multiple files."""
    print("Example 3: Model sharding")
    print("=" * 50)

    # Create a larger model
    model = create_example_model(large=True)

    # Shard the model with 5MB per shard
    # This creates:
    # - large_model.onnx
    # - large_model-00001-of-00004.safetensors
    # - large_model-00002-of-00004.safetensors
    # - large_model-00003-of-00004.safetensors
    # - large_model-00004-of-00004.safetensors
    # - large_model.safetensors.index.json (index file)
    onnx_safetensors.save_model(model, "large_model.onnx", max_shard_size="5MB")

    print("✓ Saved large_model.onnx with sharded weights")
    print("  Files created:")

    # List the created shard files
    shard_files = sorted(glob.glob("large_model-*.safetensors"))
    for shard_file in shard_files:
        size_mb = os.path.getsize(shard_file) / (1024 * 1024)
        print(f"    - {shard_file} ({size_mb:.2f} MB)")

    # Check for index file
    if os.path.exists("large_model.safetensors.index.json"):
        with open("large_model.safetensors.index.json") as f:
            index = json.load(f)
        print(f"  ✓ Index file created with {len(index['weight_map'])} tensors mapped")

    # Verify the sharded model works with ONNX Runtime
    sess = ort.InferenceSession("large_model.onnx", providers=["CPUExecutionProvider"])
    input_data = np.random.randn(1, 1000).astype(np.float32)
    outputs = sess.run(None, {"input": input_data})

    print("✓ Sharded model runs successfully with ONNX Runtime")
    print(f"  Output shape: {outputs[0].shape}")
    print()


def example_save_file_with_sharding():
    """Example 4: Use save_file with sharding for more control."""
    print("Example 4: save_file with sharding")
    print("=" * 50)

    model = create_example_model(large=True)

    # Save only the weights with sharding
    # Note: This doesn't save the ONNX model file itself
    onnx_safetensors.save_file(
        model,
        "weights_only.safetensors",
        base_dir=".",
        max_shard_size="5MB",
        replace_data=False,  # Don't modify the model
    )

    print("✓ Saved sharded weights without modifying the model")

    shard_files = sorted(glob.glob("weights_only-*.safetensors"))
    print(f"  Created {len(shard_files)} shard files")
    print()


if __name__ == "__main__":
    print("ONNX-Safetensors: save_model and Sharding Examples")
    print("=" * 50)
    print()

    # Run all examples
    example_basic_save_model()
    example_custom_weights_file()
    example_model_sharding()
    example_save_file_with_sharding()

    print("All examples completed successfully! ✓")
    print()
    print("Note: This example created several files for demonstration.")
    print("You can safely delete them after reviewing.")
