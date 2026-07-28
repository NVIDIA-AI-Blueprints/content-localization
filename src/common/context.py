# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared context for standalone client execution."""


class LocalContextAbortError(RuntimeError):
    """Raised by :meth:`LocalContext.abort` to signal a local abort.

    Subclasses ``RuntimeError`` so existing callers that catch
    ``RuntimeError`` keep working unchanged, while building the
    formatted message inside the exception class.

    Args:
        code (object): The gRPC status code (informational only).
        msg (str): The error message.

    Examples:
        >>> try:
        ...     raise LocalContextAbortError(code="INTERNAL", msg="boom")
        ... except RuntimeError as e:
        ...     print(e)
        Aborted: INTERNAL, boom
    """

    def __init__(self, *, code: object, msg: str) -> None:
        super().__init__(f"Aborted: {code}, {msg}")


class LocalContext:
    """Minimal gRPC-compatible context for local client execution.

    Provides a ``context.abort()`` method that raises
    :class:`LocalContextAbortError` (a ``RuntimeError`` subclass)
    instead of performing a real gRPC abort.  Used by all standalone
    client scripts that run outside a gRPC servicer.

    Examples:
        >>> ctx = LocalContext()
        >>> try:
        ...     ctx.abort("INTERNAL", "something went wrong")
        ... except RuntimeError as e:
        ...     print(e)
        Aborted: INTERNAL, something went wrong
    """

    def abort(self, code: object, msg: str) -> None:
        """Abort the current operation by raising an error.

        Args:
            code (object): The gRPC status code (informational only).
            msg (str): The error message.

        Returns:
            None: This method never returns normally — it always raises.

        Raises:
            LocalContextAbortError: Always raised with the provided code
                and message (a ``RuntimeError`` subclass).

        Examples:
            >>> ctx = LocalContext()
            >>> try:
            ...     ctx.abort("INTERNAL", "test error")
            ... except RuntimeError as e:
            ...     print(e)
            Aborted: INTERNAL, test error
        """
        raise LocalContextAbortError(code=code, msg=msg)
