# onnx-safetensors

[![CI](https://github.com/justinchuby/onnx-safetensors/actions/workflows/main.yml/badge.svg)](https://github.com/justinchuby/onnx-safetensors/actions/workflows/main.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/onnx-safetensors.svg)](https://pypi.org/project/onnx-safetensors)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/onnx-safetensors.svg)](https://pypi.org/project/onnx-safetensors)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Ruff](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

ONNX extension for saving to and loading from safetensors 🤗.

## Features

- ✅ Load and save ONNX weights from and to safetensors
- ✅ Support all ONNX data types, including float8, float4 and 4-bit ints
- ✅ Allow ONNX backends (including ONNX Runtime) to use safetensors

## Install

```sh
pip install --upgrade onnx-safetensors
```

## Usage

### Load tensors to an ONNX model

> [!TIP]
> You can use safetensors as external data for ONNX.

```python
import os
import onnx
import onnx_safetensors

# Provide your ONNX model here
model: onnx.ModelProto

tensor_file = "path/to/onnx_model/model.safetensors"
base_dir = "path/to/onnx_model"
data_path = "model.safetensors"

# Apply weights from the safetensors file to the model and turn them to in memory tensor
# NOTE: If model size becomes >2GB you will need to offload weights with onnx_safetensors.save_file, or onnx.save with external data options to keep the onnx model valid
model = onnx_safetensors.load_file(model, tensor_file)

# If you want to use the safetensors file in ONNX Runtime:
# Use safetensors as external data in the ONNX model
model_with_external_data = onnx_safetensors.load_file_as_external_data(model, data_path, base_dir=base_dir)

# Save the modified model
# This model is a valid ONNX model using external data from the safetensors file
onnx.save(model_with_external_data, os.path.join(base_dir, "model_using_safetensors.onnx"))
```

### Save weights to a safetensors file

```python
import onnx
import onnx_safetensors

# Provide your ONNX model here
model: onnx.ModelProto
base_dir = "path/to/onnx_model"
data_path = "model.safetensors"

# Offload weights from ONNX model to safetensors file without changing the model
onnx_safetensors.save_file(model, data_path, base_dir=base_dir, replace_data=False)  # Generates model.safetensors

# If you want to use the safetensors file in ONNX Runtime:
# Offload weights from ONNX model to safetensors file and use it as external data for the model by setting replace_data=True
model_with_external_data = onnx_safetensors.save_file(model, data_path, base_dir=base_dir, replace_data=True)

# Save the modified model
# This model is a valid ONNX model using external data from the safetensors file
onnx.save(model_with_external_data, os.path.join(base_dir, "model_using_safetensors.onnx"))
```

## Examples

- [Tutorial notebook](examples/tutorial.ipynb)
