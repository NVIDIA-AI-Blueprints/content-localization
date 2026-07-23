# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protobuf packing helpers.

Wraps native Python values in ``google.protobuf.Any`` messages using the
standard wrapper types, for APIs that accept free-form key/value
parameters.
"""

from google.protobuf import any_pb2
from google.protobuf import wrappers_pb2

# Bounds of a signed 32-bit integer; values outside this range pack as
# Int64Value instead of Int32Value.
_INT32_MAX = 2147483647
_INT32_MIN = -2147483648


def create_protobuf_any_value(value: bool | int | float | str) -> any_pb2.Any:
    """Create a ``google.protobuf.Any`` message from a Python value.

    Values map to the standard wrapper types: ``BoolValue`` for bool,
    ``Int32Value``/``Int64Value`` for int (chosen by range), and
    ``StringValue`` for str. Python floats are double precision, so they
    pack as ``DoubleValue`` to preserve their full precision.

    Args:
        value (bool | int | float | str): The value to convert.

    Returns:
        any_pb2.Any: The packed message.

    Raises:
        TypeError: If the value type is not supported.

    Examples:
        >>> packed = create_protobuf_any_value(3.14)
        >>> packed.Is(wrappers_pb2.DoubleValue.DESCRIPTOR)
        True
    """
    any_message = any_pb2.Any()

    if isinstance(value, bool):
        wrapper = wrappers_pb2.BoolValue(value=value)
        any_message.Pack(wrapper)
    elif isinstance(value, int):
        if value > _INT32_MAX or value < _INT32_MIN:
            wrapper = wrappers_pb2.Int64Value(value=value)
        else:
            wrapper = wrappers_pb2.Int32Value(value=value)
        any_message.Pack(wrapper)
    elif isinstance(value, float):
        wrapper = wrappers_pb2.DoubleValue(value=value)
        any_message.Pack(wrapper)
    elif isinstance(value, str):
        wrapper = wrappers_pb2.StringValue(value=value)
        any_message.Pack(wrapper)
    else:
        raise TypeError(f"Unsupported type: {type(value)}")

    return any_message
