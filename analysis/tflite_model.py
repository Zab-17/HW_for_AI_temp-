"""A small read-only view of a .tflite file, so analysis scripts never touch the raw FlatBuffers schema."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tensorflow.lite.python import schema_py_generated as schema

BYTES_PER_ELEMENT = {"FLOAT32": 4, "INT32": 4, "INT8": 1, "UINT8": 1, "INT16": 2, "INT64": 8}


@dataclass(frozen=True)
class TensorView:
    name: str
    shape: tuple[int, ...]
    type_name: str
    is_constant: bool

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    @property
    def size_bytes(self) -> int:
        return self.element_count * BYTES_PER_ELEMENT[self.type_name]


@dataclass(frozen=True)
class OperatorView:
    index: int
    name: str
    version: int
    fused_activation: str
    pool_window: tuple[int, int]
    inputs: tuple[TensorView, ...]
    outputs: tuple[TensorView, ...]


class TfliteModel:
    def __init__(self, model_bytes: bytes):
        self.size_bytes = len(model_bytes)
        self._model = schema.Model.GetRootAs(model_bytes, 0)
        self._graph = self._model.Subgraphs(0)

    @classmethod
    def from_file(cls, path: Path) -> TfliteModel:
        return cls(path.read_bytes())

    def operators(self) -> list[OperatorView]:
        return [self._operator(index) for index in range(self._graph.OperatorsLength())]

    def tensors(self) -> list[TensorView]:
        return [self._tensor(index) for index in range(self._graph.TensorsLength())]

    def metadata_sizes(self) -> dict[str, int]:
        """Extra named buffers the converter adds (e.g. the minimum runtime version)."""
        entries = (self._model.Metadata(i) for i in range(self._model.MetadataLength()))
        return {entry.Name().decode(): self._model.Buffers(entry.Buffer()).DataLength() for entry in entries}

    def _operator(self, index: int) -> OperatorView:
        operator = self._graph.Operators(index)
        code = self._model.OperatorCodes(operator.OpcodeIndex())
        options = _builtin_options(operator)
        return OperatorView(
            index=index,
            name=enum_name(schema.BuiltinOperator, max(code.BuiltinCode(), code.DeprecatedBuiltinCode())),
            version=code.Version(),
            fused_activation=_fused_activation(options),
            pool_window=_pool_window(options),
            inputs=tuple(self._tensor(int(i)) for i in operator.InputsAsNumpy() if i >= 0),
            outputs=tuple(self._tensor(int(i)) for i in operator.OutputsAsNumpy()),
        )

    def _tensor(self, index: int) -> TensorView:
        tensor = self._graph.Tensors(index)
        return TensorView(
            name=tensor.Name().decode(),
            shape=_shape(tensor),
            type_name=enum_name(schema.TensorType, tensor.Type()),
            is_constant=self._model.Buffers(tensor.Buffer()).DataLength() > 0,
        )


def enum_name(enum_class, value: int) -> str:
    """Schema enums are plain classes of constants; find the constant's name from its value."""
    return next(
        name for name, number in vars(enum_class).items() if not name.startswith("_") and number == value
    )


def _shape(tensor) -> tuple[int, ...]:
    """The schema returns 0 instead of an empty array for a scalar tensor."""
    dimensions = tensor.ShapeAsNumpy()
    return tuple(int(d) for d in dimensions) if isinstance(dimensions, np.ndarray) else ()


def _builtin_options(operator):
    """The operator's options table (e.g. Conv2DOptions), or a NoOptions marker for operators without one."""
    options_name = enum_name(schema.BuiltinOptions, operator.BuiltinOptionsType())
    options_class = getattr(schema, options_name, None)
    if options_class is None or operator.BuiltinOptions() is None:
        return _NO_OPTIONS
    options = options_class()
    table = operator.BuiltinOptions()
    options.Init(table.Bytes, table.Pos)
    return options


class _NoOptions:
    pass


_NO_OPTIONS = _NoOptions()


def _fused_activation(options) -> str:
    if not hasattr(options, "FusedActivationFunction"):
        return "n/a"
    return enum_name(schema.ActivationFunctionType, options.FusedActivationFunction())


def _pool_window(options) -> tuple[int, int]:
    if not hasattr(options, "FilterHeight"):
        return (0, 0)
    return (options.FilterHeight(), options.FilterWidth())
