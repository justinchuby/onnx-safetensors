from typing import Callable, Iterable
import safetensors
import onnx
import itertools


def _recursive_attribute_processor(
    attribute: onnx.AttributeProto,
    func: Callable[[onnx.GraphProto], Iterable[onnx.TensorProto]],
) -> Iterable[onnx.TensorProto]:
    """Create an iterator through processing ONNX model attributes with functor."""
    if attribute.type == onnx.AttributeProto.GRAPH:
        yield from func(attribute.g)
    if attribute.type == onnx.AttributeProto.GRAPHS:
        for graph in attribute.graphs:
            yield from func(graph)


def _get_initializer_tensors_from_graph(
    model_proto_graph: onnx.GraphProto,
) -> Iterable[onnx.TensorProto]:
    """Create an iterator of initializer tensors from ONNX model graph."""
    yield from model_proto_graph.initializer
    for node in model_proto_graph.node:
        for attribute in node.attribute:
            yield from _recursive_attribute_processor(
                attribute, _get_initializer_tensors_from_graph
            )


def _get_initializer_tensors(
    model_proto: onnx.ModelProto,
) -> Iterable[onnx.TensorProto]:
    """Create an iterator of initializer tensors from ONNX model."""
    yield from _get_initializer_tensors_from_graph(model_proto.graph)


def _get_attribute_tensors_from_graph(
    model_proto_graph: onnx.GraphProto,
) -> Iterable[onnx.TensorProto]:
    """Create an iterator of tensors from node attributes of an ONNX model graph."""
    for node in model_proto_graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors
            yield from _recursive_attribute_processor(
                attribute, _get_attribute_tensors_from_graph
            )


def _get_attribute_tensors(
    model_proto: onnx.ModelProto,
) -> Iterable[onnx.TensorProto]:
    """Create an iterator of tensors from node attributes of an ONNX model."""
    yield from _get_attribute_tensors_from_graph(model_proto.graph)


def _get_all_tensors(model_proto: onnx.ModelProto) -> Iterable[onnx.TensorProto]:
    """Scan an ONNX model for all tensors and return as an iterator."""
    return itertools.chain(
        _get_initializer_tensors(model_proto),
        _get_attribute_tensors(model_proto),
    )


def load_safetensors(model_proto: onnx.ModelProto, tensor_file: str) -> None:
    with safetensors.safe_open(tensor_file, "numpy") as f:
        keys = f.keys()
        for tensor in _get_all_tensors(model_proto):
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
