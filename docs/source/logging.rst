.. _logging:

Docker Service Logging
======================

This section describes the logging configuration for the Docker services in this project.

Overview
--------

Each Docker service writes its logs to the container's stdout/stderr, where they are captured
by Docker's default logging driver and can be viewed with ``docker logs`` or
``docker compose logs``. To collect them as local files, run
``scripts/misc/copy_docker_logs.sh``, which copies each container's logs into the ``./logs/``
directory.

Log Files
---------

The following log files are created when ``scripts/misc/copy_docker_logs.sh`` is run:

* ``./logs/s2s.log`` - Speech-to-Speech service logs
* ``./logs/lipsync.log`` - Lip Sync service logs
* ``./logs/asd.log`` - Active Speaker Detection service logs
* ``./logs/controller.log`` - Controller service logs

Log Management Scripts
----------------------

A convenience script is provided to manage log files:

* ``./scripts/misc/copy_docker_logs.sh`` - Copy logs from Docker's default location to
  ``./logs/`` directory

Usage
^^^^^

**copy_docker_logs.sh:**
.. code-block:: bash

    # Copy logs for all services
    ./scripts/misc/copy_docker_logs.sh

    # Copy logs for specific service
    ./scripts/misc/copy_docker_logs.sh s2s
    ./scripts/misc/copy_docker_logs.sh lipsync

Manual Log Access
-----------------

You can also access logs directly:

.. code-block:: bash

    # View specific service logs
    cat ./logs/s2s.log
    cat ./logs/lipsync.log

    # Follow specific service logs
    tail -f ./logs/s2s.log

    # Search logs
    rg "ERROR" ./logs/s2s.log

Docker Compose Logs
-------------------

You can still use standard Docker Compose commands to view logs:

.. code-block:: bash

    # View logs for all services
    docker compose logs

    # View logs for specific service
    docker compose logs speech-to-speech
    docker compose logs lipsync

    # Follow logs
    docker compose logs -f

Troubleshooting
---------------

If log files are not being created:

1. Run ``./scripts/misc/copy_docker_logs.sh`` — the ``./logs/`` files are only created by
   this script, not automatically by the services
2. Ensure the containers are running or exist (check with ``docker ps -a``); the script
   skips containers that do not exist
3. Ensure the current directory is writable so the script can create ``./logs/``
4. Check ``docker logs <container>`` directly to confirm the container is producing output
