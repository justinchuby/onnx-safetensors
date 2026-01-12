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

### Save an ONNX model with safetensors weights

The `save_model` function is a convenient way to save both the ONNX model and its weights to separate files:

```python
import onnx_safetensors

# Provide your ONNX model here
model: onnx.ModelProto

# Save model and weights in one step
# This creates model.onnx and model.safetensors
onnx_safetensors.save_model(model, "model.onnx")

# You can also specify a custom name for the weights file
onnx_safetensors.save_model(model, "model.onnx", external_data="weights.safetensors")
```

### Shard large models

For large models, you can automatically shard the weights across multiple safetensors files:

```python
import onnx_safetensors

# Provide your ONNX model here
model: onnx.ModelProto

# Shard the model into multiple files (e.g., 5GB per shard)
# This creates:
# - model.onnx
# - model-00001-of-00003.safetensors
# - model-00002-of-00003.safetensors
# - model-00003-of-00003.safetensors
# - model.safetensors.index.json (index file mapping tensors to shards)
onnx_safetensors.save_model(model, "model.onnx", max_shard_size="5GB")

# You can also use save_file with sharding
onnx_safetensors.save_file(
    model,
    "weights.safetensors",
    base_dir="path/to/save",
    max_shard_size="5GB"
)
```

The sharding format is compatible with the Hugging Face transformers library, making it easy to share and load models across different frameworks.

### Command Line Interface

ONNX-safetensors provides a command-line interface for converting ONNX models to use safetensors format:

```bash
# Basic conversion
onnx-safetensors convert input.onnx output.onnx

# Convert with sharding (split large models into multiple files)
onnx-safetensors convert input.onnx output.onnx --max-shard-size 5GB

# You can also specify size in MB
onnx-safetensors convert input.onnx output.onnx --max-shard-size 500MB
```

The `convert` command:

- Loads an ONNX model from the input path
- Saves it with safetensors external data to the output path
- Optionally shards large models using `--max-shard-size`
- Creates index files automatically when sharding is enabled

## Examples

- [Tutorial notebook](examples/tutorial.ipynb)
- [save_model and sharding examples](examples/save_model_sharding.py)
