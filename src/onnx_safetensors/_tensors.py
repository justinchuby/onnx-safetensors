from __future__ import annotations

import abc
import math
from typing import Any

import numpy as np
import onnx_ir as ir

from onnx_safetensors import _metadata, _type_casting


class _TensorBase(abc.ABC, ir.TensorProtocol):
    """Convenience Shared methods for classes implementing TensorProtocol."""

    __slots__ = (
        "_metadata",
        "_metadata_props",
        "doc_string",
        "name",
    )

    def __init__(
        self,
        name: str | None = None,
        doc_string: str | None = None,
        metadata_props: dict[str, str] | None = None,
    ) -> None:
        self._metadata: _metadata.MetadataStore | None = None
        self._metadata_props: dict[str, str] | None = metadata_props
        self.name: str | None = name
        self.doc_string: str | None = doc_string

    def _printable_type_shape(self) -> str:
        """Return a string representation of the shape and data type."""
        return f"{self.dtype},{self.shape}"

    def _repr_base(self) -> str:
        """Base string for the repr method.

        Example: Tensor<FLOAT,[5,42]>
        """
        return f"{self.__class__.__name__}<{self._printable_type_shape()}>"

    @property
    def size(self) -> int:
        """The number of elements in the tensor."""
        return np.prod(self.shape.numpy())  # type: ignore[return-value,attr-defined]

    @property
    def nbytes(self) -> int:
        """The number of bytes in the tensor."""
        # Use math.ceil because when dtype is INT4, the itemsize is 0.5
        return math.ceil(self.dtype.itemsize * self.size)

    @property
    def metadata_props(self) -> dict[str, str]:
        if self._metadata_props is None:
            self._metadata_props = {}
        return self._metadata_props

    @property
    def meta(self) -> _metadata.MetadataStore:
        """The metadata store for intermediate analysis.

        Write to the :attr:`metadata_props` if you would like the metadata to be serialized
        to the ONNX proto.
        """
        if self._metadata is None:
            self._metadata = _metadata.MetadataStore()
        return self._metadata


class ByteArrayTensor(_TensorBase):
    """A tensor initialized from bytes."""

    def __init__(
        self,
        data: bytearray,
        dtype: ir.DataType,
        shape: ir.Shape,
        name: str = "",
        doc_string: str = "",
    ) -> None:
        super().__init__(name=name, doc_string=doc_string)
        self.raw = data
        self._dtype = dtype
        self._shape = shape

    @property
    def shape(self) -> ir.Shape:
        return self._shape

    @property
    def dtype(self) -> ir.DataType:
        return self._dtype

    def __repr__(self) -> str:
        # It is a little hard to display the content when there can be types
        # unsupported by numpy
        # Preferably we should display some content when the tensor is small
        return f"{self._repr_base()}(name={self.name!r})"

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """Return the tensor as a numpy array, compatible with np.array."""
        return self.numpy().__array__(dtype)

    def __dlpack__(self, *, stream: Any = None) -> Any:
        return self.numpy().__dlpack__(stream=stream)

    def __dlpack_device__(self) -> tuple[int, int]:
        return self.numpy().__dlpack_device__()

    def numpy(self) -> np.ndarray:
        dtype = self.dtype
        if dtype == ir.DataType.UNDEFINED:
            raise ValueError("Cannot convert UNDEFINED tensor to numpy array.")

        array = np.frombuffer(self.raw, dtype=dtype.numpy().newbyteorder("<"))
        # Cannot return now, because we may need to unpack 4bit tensors
        shape = self._shape.numpy()
        if dtype == ir.DataType.INT4:
            return _type_casting.unpack_int4(array.astype(np.uint8), shape)
        elif dtype == ir.DataType.UINT4:
            return _type_casting.unpack_uint4(array.astype(np.uint8), shape)
        elif dtype == ir.DataType.FLOAT4E2M1:
            return _type_casting.unpack_float4e2m1(array.astype(np.uint8), shape)
        else:
            # Otherwise convert to the correct dtype and reshape
            # Note we cannot use view() here because the storage dtype may not be the same size as the target
            return array.reshape(shape)

    def tobytes(self) -> bytes:
        return bytes(self.raw)
