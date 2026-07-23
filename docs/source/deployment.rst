.. _deployment:

Deployment
==========

This section covers deploying the Content Localization Blueprint services for both development and production environments.

Docker Compose Profiles
------------------------

The blueprint uses Docker Compose profiles to configure different service combinations for various use cases.

Available Profiles
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 8 8 10 12 10 34

   * - Profile
     - S2S
     - ASD
     - LipSync
     - Controller
     - Demo App
     - Description
   * - ``default``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - All services (for testing)
   * - ``third-party-s2s``
     - ✓
     - \-
     - \-
     - \-
     - \-
     - S2S only with ElevenLabs/CambAI
   * - ``lipsync``
     - ✓
     - \-
     - ✓
     - \-
     - \-
     - LipSync + S2S backend
   * - ``third-party-s2s-lipsync``
     - ✓
     - \-
     - ✓
     - \-
     - \-
     - S2S (ElevenLabs/CambAI) + LipSync
   * - ``third-party-s2s-asd-lipsync``
     - ✓
     - ✓
     - ✓
     - \-
     - \-
     - Full pipeline with ElevenLabs/CambAI
   * - ``asd``
     - \-
     - ✓
     - \-
     - \-
     - \-
     - Active Speaker Detection only
   * - ``controller-third-party-s2s``
     - ✓
     - ✓
     - ✓
     - ✓
     - \-
     - Orchestrated pipeline (ElevenLabs/CambAI)
   * - ``asd-lipsync``
     - \-
     - ✓
     - ✓
     - ✓
     - \-
     - ASD + LipSync + Controller
   * - ``demo-app-third-party-s2s``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - Full stack with Web Demo (ElevenLabs/CambAI)

Usage Examples
~~~~~~~~~~~~~~

ElevenLabs/CambAI with full pipeline and demo app:

.. code-block:: bash

   docker compose --profile demo-app-third-party-s2s \
       --env-file configs/elevenlabs.env \
       --env-file .env \
       up --build

Controller orchestration with ElevenLabs/CambAI:

.. code-block:: bash

   docker compose --profile controller-third-party-s2s \
       --env-file configs/elevenlabs.env \
       --env-file .env \
       up --build

Profile Selection Guide
~~~~~~~~~~~~~~~~~~~~~~~

* **For Development/Testing**: Use ``demo-app-third-party-s2s`` for the full stack with web interface
* **For Production with ElevenLabs/CambAI**: Use ``controller-third-party-s2s`` for orchestrated processing
* **For Service Testing**: Use individual profiles like ``third-party-s2s``, ``lipsync``, or ``asd``

First-Time Deployment
---------------------

For first-time deployment, use the deploy scripts to verify each service individually before deploying the full stack.

Deploy LipSync Service
~~~~~~~~~~~~~~~~~~~~~~

Deploy the LipSync service:

.. code-block:: bash

   ./scripts/nims/deploy_lipsync.sh

This will:

* Download the LipSync models to ``volumes/models/lipsync/``
* Start the LipSync container on ports 8004 (HTTP) and 50054 (gRPC)
* Requires ``LIPSYNC_API_KEY`` environment variable

Deploy ASD Service
~~~~~~~~~~~~~~~~~~

Deploy the Active Speaker Detection (ASD) NIM service:

.. code-block:: bash

   ./scripts/nims/deploy_asd.sh

This will:

* Deploy ASD NIM container with GPU support
* Start ASD container on ports 8005 (HTTP) and 50055 (gRPC)
* Mount model cache at ``volumes/models/asd/``
* Requires ``ASD_API_KEY`` environment variable

Deployment Notes
~~~~~~~~~~~~~~~~

* Each deploy script runs in interactive mode (``-it``) and will occupy your terminal
* Run each script in a separate terminal or stop (Ctrl+C) before running the next
* These scripts are for **verification only** - use docker compose for production deployments

Service Management
------------------

Compose Service vs Container Names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``docker compose`` subcommands that target a single service use the **service
name** from ``docker-compose.yml``. This differs from the **container name** for
the S2S service (service ``speech-to-speech`` runs in container ``s2s``). Use the
service name with ``docker compose`` commands and the container name with plain
``docker`` commands.

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Compose service
     - Container name
     - gRPC port
   * - ``speech-to-speech``
     - ``s2s``
     - ``50050``
   * - ``asd``
     - ``asd``
     - ``50055``
   * - ``lipsync``
     - ``lipsync``
     - ``50054``
   * - ``controller``
     - ``controller``
     - ``50056``
   * - ``demo-app``
     - ``demo-app``
     - HTTP ``3000``

For example, restart S2S with ``docker compose restart speech-to-speech`` (service
name), but tail its container logs with ``docker logs s2s`` (container name).

Starting Services
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Default profile: Full stack with ElevenLabs/CambAI and demo app
   docker compose --profile demo-app-third-party-s2s \
       --env-file configs/elevenlabs.env \
       --env-file .env \
       up --build

Stopping Services
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Stop all services
   docker compose down

   # Stop and remove volumes (clean state)
   docker compose down -v

Viewing Logs
~~~~~~~~~~~~

Real-Time Log Viewing
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # View logs from all services (follow mode)
   docker compose logs -f

   # View logs from specific service
   docker compose logs -f speech-to-speech
   docker compose logs -f controller
   docker compose logs -f lipsync

   # View last 100 lines of logs
   docker compose logs --tail=100

Copy Logs to Files
^^^^^^^^^^^^^^^^^^

Use the log copy script to save logs to local files:

.. code-block:: bash

   # Copy all service logs
   ./scripts/misc/copy_docker_logs.sh

   # Copy specific service logs
   ./scripts/misc/copy_docker_logs.sh s2s
   ./scripts/misc/copy_docker_logs.sh controller

This creates log files in ``./logs/``:

* ``./logs/s2s.log`` - Speech-to-Speech service logs
* ``./logs/lipsync.log`` - LipSync service logs
* ``./logs/asd.log`` - Active Speaker Detection logs
* ``./logs/controller.log`` - Controller orchestration logs

Benefits of Copying Logs
^^^^^^^^^^^^^^^^^^^^^^^^^

* Persist logs even after containers are stopped
* Easy to share with team members for debugging
* Can be archived or uploaded to issue trackers
* Includes line counts and helpful status messages
