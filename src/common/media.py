# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Media file inspection helpers.

Validates media input files (existence and extension checks) and
inspects MP4 atom layout to determine whether a file can be streamed
progressively.
"""

import os

# MP4 atoms start with a fixed 8-byte header: a 4-byte big-endian size
# followed by a 4-byte type tag.
_ATOM_HEADER_SIZE = 8


def is_file_available(file_path: os.PathLike, file_types: list[str]) -> bool:
    """Check if the file exists and has one of the specified file types.

    Args:
        file_path (os.PathLike): Path to the input file.
        file_types (list[str]): List of allowed file extensions (without
            the dot, e.g., ``["mp4", "wav"]``).

    Returns:
        bool: ``True`` if the file exists and has one of the allowed
            extensions, ``False`` otherwise.

    Raises:
        FileNotFoundError: If the file does not exist.

    Examples:
        >>> is_file_available("input.mp4", ["mp4"])  # doctest: +SKIP
        True
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found")
    for file_type in file_types:
        if os.path.splitext(file_path)[1].lower() == f".{file_type}":
            return True
    return False


def check_streamable(file_path: os.PathLike) -> bool:
    """Checks if the video is streamable by checking if the moov atom follows immediately after
    the ftyp atom in an MP4 file.

    For streamable MP4s, the moov atom must come immediately after:
    [4 bytes: size][4 bytes: "ftyp"][... ftyp data ...]
    [4 bytes: size][4 bytes: "moov"][... moov data ...]

    For non-streamable MP4s, other atoms like mdat may come between ftyp and moov:
    [4 bytes: size][4 bytes: "ftyp"][... ftyp data ...][4 bytes: size][4 bytes: "mdat"]
    [... mdat data ...][moov atom]

    Args:
        file_path (os.PathLike): Path to the MP4 file to check.

    Returns:
        bool: ``True`` if the file is streamable, ``False`` otherwise.

    Examples:
        >>> check_streamable("input.mp4")  # doctest: +SKIP
        True
    """
    with open(file_path, "rb") as f:
        # First atom header: [4 bytes: size][4 bytes: type]
        atom_header = f.read(_ATOM_HEADER_SIZE)
        if len(atom_header) < _ATOM_HEADER_SIZE:
            return False

        ftyp_size = int.from_bytes(atom_header[0:4], byteorder="big")
        # A size below the header size cannot describe a complete atom
        # (size 1 denotes a 64-bit extended size, which never applies to
        # ftyp in practice).
        if atom_header[4:8] != b"ftyp" or ftyp_size < _ATOM_HEADER_SIZE:
            return False

        # The ftyp atom size varies with the number of compatible brands and
        # is untrusted file data, so skip the remaining ftyp body with a seek
        # instead of reading it into memory, then read only the fixed-size
        # header of the following atom.
        f.seek(ftyp_size - _ATOM_HEADER_SIZE, os.SEEK_CUR)
        next_header = f.read(_ATOM_HEADER_SIZE)
        if len(next_header) < _ATOM_HEADER_SIZE:
            # The following atom header is absent or truncated (seeking past
            # EOF yields a short read here).
            return False

    # The type tag occupies the last 4 bytes of the 8-byte atom header.
    return next_header[4:8] == b"moov"
