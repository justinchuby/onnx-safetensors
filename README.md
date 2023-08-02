# onnx-safetensors
ONNX extension for saving to and loading from safetensors

## Install

```
pip install --upgrade onnx-safetensors
```

## Usage

Load tensors to an ONNX model

```python
import onnx_safetensors

# Provide your ONNX model here
model: onnx.ModelProto
tensor_file = "model.safetensors"
# Apply weights from the safetensors file to the model
onnx_safetensors.load_file(model, tensor_file)
```

Save weights to a safetensors file

```python
import onnx_safetensors

# Provide your ONNX model here
model: onnx.ModelProto
tensor_file = "model.safetensors"
# Save weights from to the safetensors file
onnx_safetensors.save_file(model, tensor_file, convert_attributes=True)

# Save weights from to the safetensors file and clear the raw_data fields of the ONNX model to reduce its size
# model will be updated inplace
onnx_safetensors.save_file(model, tensor_file, convert_attributes=True, strip_data=True)
```
