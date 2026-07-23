# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path

current_file_path = Path(__file__)
project_root = Path(current_file_path).parent.parent.parent
sys.path[:0] = [
    str(project_root),
    str(project_root / "src"),
    str(project_root / "client"),
    str(project_root / "protos" / "generated"),
]
print(f"{sys.path=}")

# Mock problematic modules before importing s2s_service to avoid import errors during doc build
from unittest.mock import MagicMock

# Mock modules that might not be available or cause issues during doc build
MOCK_MODULES = [
    # "cv2",
    # "numpy",
    # "numpy.linalg",
    # "numpy.ndarray",
    "onnxruntime",
    "skimage",
    "skimage.transform",
    "sklearn",
    "tensorflow",
    "torch",
    "torchvision",
    # "PIL",
    # "PIL.Image",
    # "PIL.ImageDraw",
    # "PIL.ImageFont",
    # "PIL.PngImagePlugin",
    # "mediapipe",
    # "face_recognition",
    # "dlib",
    "librosa",
    "pydub",
    # scipy's internal initialization can break with certain numpy
    # versions; mock it so doc builds stay resilient.
    "scipy",
    "scipy.signal",
    "scipy.io",
    "scipy.io.wavfile",
    # "moviepy",
    # "moviepy.editor",
    # "ffmpeg",
    # "ffmpeg_python",
    "gi",
    "gi.repository",
    ## matplotlib modules (client-side latency plotting; a dev-only dependency
    ## that is not installed in the docs build environment, so it is mocked)
    "matplotlib",
    "matplotlib.pyplot",
    # "matplotlib.figure",
    # "matplotlib.axes",
    # "traceback",
    # "itertools",
    ## ElevenLabs modules
    "elevenlabs",
    "elevenlabs.client",
    ## our AI4M modules
    # "ai4m_base_utils",
    # "ai4m_base_utils.grpc_service",
    # "ai4m_base_utils.grpc_service.GRPCServiceBase",
    # "ai4m_base_utils.config",
    # "ai4m_base_utils.logger",
    # "ai4m_base_utils.gpu_utils",
    # "ai4m_base_utils.file_utils",
    ## CUDA and TensorRT modules
    # "pycuda",
    # "pycuda.driver",
    # "pycuda.autoinit",
    # "tensorrt",
    # "tensorrt.tensorrt",
    # "cvcuda",
    # "vpf",
]

# Create comprehensive mocks for problematic modules
for module_name in MOCK_MODULES:
    mock = MagicMock()
    # Special handling for logger to avoid comparison errors
    if module_name == "ai4m_base_utils.logger":
        mock.getEffectiveLevel.return_value = 20  # INFO level
    # Special handling for grpc_service to avoid metaclass conflicts
    elif module_name == "ai4m_base_utils.grpc_service":
        # Create a proper base class mock that can be inherited from
        class MockGRPCServiceBase:
            def __init__(self, *args, **kwargs):
                pass

            @staticmethod
            def argsfactory(parser):
                return parser

        mock.GRPCServiceBase = MockGRPCServiceBase
    sys.modules[module_name] = mock

# Ensure the logger mock is properly configured
if "ai4m_base_utils.logger" in sys.modules:
    logger_mock = sys.modules["ai4m_base_utils.logger"]
    logger_mock.getEffectiveLevel.return_value = 20  # INFO level
    # Also mock the logger object that gets imported
    logger_mock.logger = MagicMock()
    logger_mock.logger.getEffectiveLevel.return_value = 20  # INFO level

# Special handling for numpy to support union types
if "numpy" in sys.modules:
    numpy_mock = sys.modules["numpy"]
    numpy_mock.ndarray = MagicMock()
    numpy_mock.linalg = MagicMock()

# Mock environment variables
os.environ["ELEVENLABS_API_KEY"] = "dummy"

import client
import controller_service
import s2s_service

# Don't import submodules directly - let Sphinx handle them through autodoc
# The mocking above should be sufficient for Sphinx to process the modules


print(f"Building docs for s2s service version: {s2s_service.__version__}")
print(f"Building docs for client version: {client.__version__}")
print("Building docs with external ASD NIM integration")
print(f"Building docs for controller service version: {controller_service.__version__}")

project = "Content Localization Blueprint"
copyright = "2026, NVIDIA"  # noqa: A001
author = "NVIDIA"
release = "1.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.mermaid",
]

html_use_smartypants = True
todo_include_todos = True
autosummary_generate = False
autosummary_imported_members = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

# Exclude system library modules from autodoc to avoid indentation errors
autodoc_mock_imports = [
    "contextlib",
    "abc",
    "dataclasses",
    "pathlib",
    "queue",
    "builtins",
    *MOCK_MODULES,
]

# Autodoc configuration to handle duplicates
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}

autodoc_warningiserror = False

# Don't show full module paths in documentation
add_module_names = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "docs/manuals/py/**",
    "uml_mermaid/README.rst",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#000000",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    # 'navigation_depth': 10,
    "includehidden": True,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_logo = os.path.join("../theme", "nv_logo.png")
html_favicon = os.path.join("../theme", "Nvidia.ico")
html_last_updated_fmt = ""
html_title = project
html_use_index = True
html_static_path = ["_static"]

html_css_files = []
html_js_files = []

# Disable the ELK layout loader — @mermaid-js/layout-elk@0.1.4 silently
# breaks mermaid initialization, preventing all diagrams from rendering.
# Standard dagre layout works fine for all diagrams in this project.
mermaid_include_elk = False


def setup(app):
    app.add_css_file("css/custom.css")
    app.add_js_file("js/pk_scripts.js")
