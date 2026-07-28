.. _setup:

Development Environment Setup
==============================

This guide walks through setting up your local development environment for the Content Localization Blueprint.

Prerequisites
-------------

Before setting up the development environment, ensure you have the following installed.

System Requirements
~~~~~~~~~~~~~~~~~~~

* **Operating System**: Linux (Ubuntu 22.04 or 24.04 recommended)
* **Python**: 3.12 or higher
* **Git**: With Git LFS enabled
* **NVIDIA GPU**: With CUDA-capable drivers installed
* **CUDA Toolkit**: CUDA 12.x
* **TensorRT**: Compatible with your CUDA version
* **Docker**: Docker Engine 24.x or higher with Docker Compose and Nvidia docker runtime
* **Hardware**: For all services to be hosted on the same GPU, a GPU of 32GB memory or higher is recommended. Refer to each NIM for their support matrices.

Verify GPU and CUDA Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Verify NVIDIA driver installation
   nvidia-smi

   # Check CUDA version
   nvcc --version

Install System Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y \
       curl \
       wget \
       git-lfs

Required Credentials
~~~~~~~~~~~~~~~~~~~~

You need the following credentials set as environment variables (or in a ``.env`` file).
Get your NGC API keys from: https://ngc.nvidia.com/setup/api-key

**NIM container keys** — ``docker-compose.yml`` maps these component-specific keys to
``NGC_API_KEY`` inside each NIM container. Set the keys for the services you intend to run:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Variable
     - Used by
     - When required
   * - ``LIPSYNC_API_KEY``
     - LipSync NIM container
     - When using LipSync
   * - ``ASD_API_KEY``
     - ASD NIM container
     - When using Active Speaker Detection

**Third-party API keys** — required only when using the corresponding S2S backend:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - When required
   * - ``ELEVENLABS_API_KEY``
     - When ``S2S_SERVICE=EL_DUBBING``
   * - ``CAMB_API_KEY``
     - When ``S2S_SERVICE=CAMB_DUBBING`` (and for the CAMB.AI helper
       scripts)

Setup Steps
-----------

1. Create Environment File
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a ``.env`` file in the project root with your credentials:

.. code-block:: bash

   # .env file

   # NIM container keys (mapped to NGC_API_KEY inside each container by docker-compose.yml)
   LIPSYNC_API_KEY=your_ngc_api_key_here
   ASD_API_KEY=your_ngc_api_key_here

   # Third-party S2S backend keys
   ELEVENLABS_API_KEY=your_11labs_api_key_here
   # CAMB_API_KEY=your_camb_api_key_here

2. Install uv Package Manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install ``uv``, the fast Python package manager:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

Add ``uv`` to your PATH (or restart your shell):

.. code-block:: bash

   export PATH="$HOME/.local/bin:$PATH"

3. Create Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create and activate a Python 3.12 virtual environment:

.. code-block:: bash

   uv venv --python 3.12
   source .venv/bin/activate

4. Install Python Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the project dependencies and commonly used extras:

.. code-block:: bash

   # Install core dependencies with non-GPU extras (test, lint, docs)
   uv sync --extra test --extra lint --extra docs

Install GPU extras only on hosts with CUDA Toolkit headers available (``cuda.h``):

.. code-block:: bash

   # Optional: install GPU extras (requires CUDA Toolkit development headers)
   uv sync --extra gpu

5. Install Development Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install linting and pre-commit hooks:

.. code-block:: bash

   uv tool install pre-commit
   uv tool install ruff
   pre-commit install

6. Generate Protocol Buffer Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate Python code from the gRPC/protobuf definitions:

.. code-block:: bash

   # Download gRPC health check proto
   wget -O protos/health.proto https://raw.githubusercontent.com/grpc/grpc/master/src/proto/grpc/health/v1/health.proto

   # Generate Python protobuf files
   bash ./protos/generate_protos.sh

7. Set Python Path
~~~~~~~~~~~~~~~~~~

Add the project root, ``src``, ``client``, and generated protobuf files to your Python path:

.. code-block:: bash

   export PYTHONPATH="${PYTHONPATH}:${PWD}:${PWD}/src:${PWD}/client:${PWD}/protos/generated"

Add this line to your shell profile (``.bashrc`` or ``.zshrc``) to make it permanent:

.. code-block:: bash

   echo 'export PYTHONPATH="${PYTHONPATH}:'"${PWD}"':'"${PWD}"'/src:'"${PWD}"'/client:'"${PWD}"'/protos/generated"' >> ~/.bashrc

8. Create Required Directories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create directories for test outputs and builds:

.. code-block:: bash

   mkdir -p build
   mkdir -p outputs

Required API Keys
-----------------

Before deploying services via ``docker compose``, set the relevant API keys
in a local ``.env`` file. ``docker-compose.yml`` maps each component-specific
key to ``NGC_API_KEY`` inside the corresponding NIM container, so setting
only a top-level ``NGC_API_KEY`` is **not** sufficient. Only the keys for
the services you intend to run are required:

.. code-block:: ini

   # LipSync NIM (mapped to NGC_API_KEY inside the lipsync container)
   LIPSYNC_API_KEY=<your_ngc_api_key>

   # Active Speaker Detection (mapped to NGC_API_KEY inside the ASD container)
   ASD_API_KEY=<your_ngc_api_key>

   # ElevenLabs backend (passed directly to S2S service)
   ELEVENLABS_API_KEY=<your_elevenlabs_api_key>

Automated Setup
---------------

For automated setup, use the provided script:

.. code-block:: bash

   # Full setup including Docker and GPU drivers
   ./scripts/misc/setup_env.sh

   # Development setup (adds lint and pre-commit tools)
   ./scripts/misc/setup_env.sh --dev

   # Skip Docker and GPU driver installation
   ./scripts/misc/setup_env.sh --no-docker --no-gpu --dev

This script automatically performs all the setup steps above.

Verify Setup
------------

If all steps completed successfully, you're ready to run the full service stack!

To verify your setup:

1. Run the test suite: ``source .venv/bin/activate && python -m pytest tests/``
2. Run the linter: ``source .venv/bin/activate && ruff check src/ tests/ client/``
3. Build the documentation: ``bash docs/build_docs.sh``

You can now proceed to the :ref:`deployment` section to launch services with docker compose.
