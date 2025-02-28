import numpy as np
import onnx
from onnx import TensorProto, helper

# Create a simple model: Y = X + W, where W is an initializer
input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

# Create an initializer tensor
weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
weights_initializer = helper.make_tensor(
    name="weights",
    data_type=TensorProto.FLOAT,
    dims=weights.shape,
    vals=weights.flatten().tolist(),
)

# Create a node (Add operation)
node_def = helper.make_node(
    "Add",
    inputs=["input", "weights"],
    outputs=["output"],
)

# Create the graph
graph_def = helper.make_graph(
    nodes=[node_def],
    name="SimpleGraph",
    inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])],
    outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
    initializer=[weights_initializer],
)

# Create the model
model_def = helper.make_model(graph_def, producer_name="onnx-safetensors-example")

# Save the model
onnx.save(model_def, "model.textproto")
