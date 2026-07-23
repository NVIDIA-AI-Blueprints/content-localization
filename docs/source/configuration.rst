.. _configuration:

=============
Configuration
=============

All services are configured via environment variables set in
``configs/*.env`` files and passed through ``docker-compose.yml``.
Secrets (API keys) go in a ``.env`` file at the repo root, which is
git-ignored.

Configuration Files
-------------------

Three profiles live in the ``configs/`` directory:

- ``configs/elevenlabs.env`` — ElevenLabs S2S backend
  (``S2S_SERVICE=EL_DUBBING``).
- ``configs/camb.env`` — CambAI S2S backend
  (``S2S_SERVICE=CAMB_DUBBING``).
- ``configs/debug.env`` — developer profile. Identical knobs to
  ``configs/elevenlabs.env`` with debug-oriented values
  (``LIPSYNC_DEBUG_MODE=1``).

**Timeouts and intervals:** All environment variables that specify a
timeout or interval (e.g. ``HEALTH_CHECK_TIMEOUT``, ``BUFFER_POLL_TIMEOUT``,
``CONTROLLER_CLEANUP_TIMEOUT``,
``S2S_EL_DUBBING_POLL_INTERVAL``, ``S2S_EL_KEEPALIVE_INTERVAL``) use
**seconds** unless otherwise documented.

Shared Configuration
--------------------

These variables are used by **both** the Controller and S2S services.

.. list-table::
   :header-rows: 1
   :widths: 35 10 10 45

   * - Variable
     - Type
     - Default
     - Description
   * - ``HEALTH_CHECK_TIMEOUT``
     - float
     - ``5.0``
     - Seconds for HTTP and gRPC health-check requests before timeout.
   * - ``BUFFER_POLL_TIMEOUT``
     - float
     - ``0.1``
     - Seconds between poll attempts in ``RequestIteratorFromBuffer``.
       Controls how often buffer-backed iterators check for new items
       or exhaustion.

Controller Service
------------------

Basic
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``CONTROLLER_GRPC_API_PORT``
     - int
     - ``50056``
     - gRPC listen port.
   * - ``CONTROLLER_MAX_CONCURRENCY``
     - int
     - ``1``
     - Maximum concurrent requests.
   * - ``CONTROLLER_LOG_LEVEL``
     - str
     - ``INFO``
     - Logging level.
   * - ``CONTROLLER_GRPC_CONCURRENCY_MODE``
     - str
     - ``threading``
     - gRPC concurrency mode (``threading`` or ``multiprocessing``).
   * - ``CONTROLLER_GRPC_THREADS_PER_PROCESS``
     - int
     - ``1``
     - Worker threads per gRPC process.

Service Endpoints
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 10 15 45

   * - Variable
     - Type
     - Default
     - Description
   * - ``S2S_SERVER``
     - str
     - (set by compose)
     - S2S service ``host:port``. Optional when running in
       bypass-S2S-only mode.
   * - ``ASD_SERVER``
     - str
     - (set by compose)
     - ASD NIM service ``host:port``. Not required when
       ``bypass_asd=True`` in ``ContentLocalizationConfig``.
   * - ``LIPSYNC_SERVER``
     - str
     - (set by compose)
     - LipSync NIM service ``host:port``.

Processing and Timeouts
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``S2S_SERVICE``
     - str
     - ``EL_DUBBING``
     - S2S backend (``EL_DUBBING`` or ``CAMB_DUBBING``).
   * - ``CONTROLLER_CONFIG_POLL_TIMEOUT``
     - float
     - ``5.0``
     - Seconds to wait for per-request config messages
       (``controller_config``, ``asd_config``, ``lipsync_config``)
       before treating them as absent.
   * - ``CONTROLLER_CLEANUP_TIMEOUT``
     - float
     - ``10.0``
     - Seconds to wait for the deserializer thread and client threads
       to finish during per-request cleanup.

Transport Security (TLS)
~~~~~~~~~~~~~~~~~~~~~~~~

TLS is available on the hops whose servers can terminate it: the
controller's serving endpoint, the in-repo S2S service, and the
controller's downstream connections (per hop). The ASD and LipSync NIM
images expose no TLS surface, so those hops require a TLS-terminating
proxy or service mesh. By design, all hops run plaintext by default
inside the trusted docker-compose bridge network. For deployments whose
service-to-service traffic crosses untrusted networks, enable TLS per
hop:

- **Server side** (client → controller, controller → S2S): pass the CLI
  flags provided by the gRPC service base — ``--use-ssl``,
  ``--ssl_server_key_path``, ``--ssl_server_cert_path``, and (for mTLS)
  ``--ssl_root_cert_path`` — to ``src/controller_service/entrypoint.py``
  or ``src/s2s_service/entrypoint.py``.
- **Downstream side** (controller → S2S/ASD/LipSync): use the
  ``--ssl-mode``/``--ssl-key``/``--ssl-cert``/``--ssl-root-cert`` flags or
  the environment variables below (flags take precedence, mirroring the
  client apps' ``--ssl-mode`` surface).

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``CONTROLLER_NIM_SSL_MODE``
     - str
     - ``DISABLED``
     - Channel security for downstream NIM connections:
       ``DISABLED``, ``TLS``, or ``MTLS``.
   * - ``CONTROLLER_NIM_SSL_KEY``
     - str
     - (unset)
     - Path to the client private key PEM (required for ``MTLS``).
   * - ``CONTROLLER_NIM_SSL_CERT``
     - str
     - (unset)
     - Path to the client certificate chain PEM (required for ``MTLS``).
   * - ``CONTROLLER_NIM_SSL_ROOT_CERT``
     - str
     - (unset)
     - Path to the root certificate PEM used to verify downstream servers
       (required for ``TLS`` and ``MTLS``).
   * - ``CONTROLLER_S2S_SSL_MODE``
     - str
     - (unset)
     - Per-hop override of ``CONTROLLER_NIM_SSL_MODE`` for the S2S
       connection (``DISABLED``, ``TLS``, or ``MTLS``). Unset inherits
       the global mode.
   * - ``CONTROLLER_ASD_SSL_MODE``
     - str
     - (unset)
     - Per-hop override of ``CONTROLLER_NIM_SSL_MODE`` for the ASD
       connection. Unset inherits the global mode.
   * - ``CONTROLLER_LIPSYNC_SSL_MODE``
     - str
     - (unset)
     - Per-hop override of ``CONTROLLER_NIM_SSL_MODE`` for the LipSync
       connection. Unset inherits the global mode.

.. warning::

   A hop's mode must match what its server actually terminates. The
   in-repo Speech-to-Speech service serves plaintext unless started
   with ``--use-ssl``/``--ssl_server_key_path``/``--ssl_server_cert_path``
   (the same server-side surface the controller exposes); the ASD and
   LipSync NIM images expose no TLS surface, so those hops must stay
   ``DISABLED`` unless fronted by a TLS-terminating proxy or service
   mesh. Development certificates for testing this surface can be
   generated with ``scripts/misc/generate_dev_certs.sh``.

.. note::

   The docker-compose healthchecks probe each service with
   ``grpcurl --plaintext``. When a service is started with ``--use-ssl``,
   override its healthcheck to match — replace ``--plaintext`` with
   ``-cacert <root-cert.pem>`` (or ``-insecure`` for self-signed
   development certificates) — otherwise the container is reported
   unhealthy even though the service is serving TLS correctly.

Debug
~~~~~

The debugpy port is only published when the opt-in override file
``docker-compose.debug.yml`` is added to the compose invocation
(``docker compose -f docker-compose.yml -f docker-compose.debug.yml ...``);
the production image does not ship debugpy — the entrypoint installs it at
container start when ``CONTROLLER_VS_CODE_DEBUG=1``.

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``CONTROLLER_DEBUG_PORT``
     - int
     - ``5678``
     - VS Code debugpy listen port.
   * - ``CONTROLLER_VS_CODE_DEBUG``
     - int
     - ``0``
     - Set to ``1`` to enable debugpy wait-for-client mode.

S2S Service
-----------

Basic
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``S2S_GRPC_API_PORT``
     - int
     - ``50050``
     - gRPC listen port.
   * - ``S2S_MAX_CONCURRENCY``
     - int
     - ``1``
     - Maximum concurrent requests.
   * - ``S2S_LOG_LEVEL``
     - str
     - ``INFO``
     - Logging level.
   * - ``S2S_GRPC_CONCURRENCY_MODE``
     - str
     - ``threading``
     - gRPC concurrency mode (``threading`` or ``multiprocessing``).
   * - ``S2S_GRPC_THREADS_PER_PROCESS``
     - int
     - ``1``
     - Worker threads per gRPC process.
   * - ``S2S_SAMPLE_RATE_HZ``
     - int
     - ``16000``
     - Input audio sample rate (Hz).
   * - ``S2S_MESSAGE_SIZE``
     - int
     - ``67108864``
     - Maximum gRPC message size (bytes).

Language
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``S2S_DEFAULT_SOURCE_LANGUAGE``
     - str
     - ``auto``
     - Source language code (``auto`` for ElevenLabs auto-detect).
   * - ``S2S_DEFAULT_TARGET_LANGUAGE``
     - str
     - Backend-specific
     - Target language code (``de`` in ``configs/elevenlabs.env``, ``26`` in
       ``configs/camb.env``).

ElevenLabs
~~~~~~~~~~

These apply only when ``S2S_SERVICE=EL_DUBBING``.

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``S2S_EL_DUBBING_POLL_INTERVAL``
     - int
     - ``10``
     - Seconds between ElevenLabs dubbing status checks.
   * - ``S2S_EL_DUBBING_MAX_ATTEMPTS``
     - int
     - ``120``
     - Maximum dubbing status poll attempts before timeout
       (total max wait = interval x attempts, default 20 minutes).
   * - ``S2S_EL_KEEPALIVE_INTERVAL``
     - int
     - ``1``
     - Seconds between keepalive pings sent to the client while
       waiting for ElevenLabs dubbing to complete.

CambAI
~~~~~~

These apply only when ``S2S_SERVICE=CAMB_DUBBING``.

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``S2S_CAMB_DUBBING_POLL_INTERVAL``
     - int
     - ``10``
     - Seconds between CambAI dubbing status checks.
   * - ``S2S_CAMB_DUBBING_MAX_ATTEMPTS``
     - int
     - ``120``
     - Maximum dubbing status poll attempts before timeout
       (total max wait = interval x attempts, default 20 minutes).
   * - ``S2S_CAMB_KEEPALIVE_INTERVAL``
     - int
     - ``1``
     - Seconds between keepalive pings sent to the client while
       waiting for CambAI dubbing to complete.
   * - ``S2S_CAMB_ALT_FORMAT_POLL_INTERVAL``
     - int
     - ``S2S_CAMB_DUBBING_POLL_INTERVAL``
     - Seconds between CambAI alt-format (MP3) output status checks.
       Falls back to ``S2S_CAMB_DUBBING_POLL_INTERVAL`` (default 10)
       when unset.
   * - ``S2S_CAMB_ALT_FORMAT_MAX_ATTEMPTS``
     - int
     - ``S2S_CAMB_DUBBING_MAX_ATTEMPTS``
     - Maximum alt-format status poll attempts before timeout. Falls
       back to ``S2S_CAMB_DUBBING_MAX_ATTEMPTS`` (default 120) when
       unset.

ASD NIM
-------

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``ASD_IMAGE``
     - str
     - (set in ``configs/*.env``)
     - ASD NIM container image reference used by
       ``docker-compose.yml``.
   * - ``ASD_GRPC_API_PORT``
     - int
     - ``50055``
     - gRPC listen port.
   * - ``ASD_NIM_HTTP_API_PORT``
     - int
     - ``8005``
     - HTTP API port (health endpoint) exposed by the ASD NIM
       container.
   * - ``ASD_MODEL_MOUNT_PATH``
     - str
     - ``./volumes/models/asd``
     - Host path mounted as the ASD NIM model cache
       (``/opt/nim/.cache``).
   * - ``ASD_DIARIZATION_TOLERANCE_MS``
     - int
     - ``0``
     - Diarization timestamp tolerance (milliseconds) passed through
       to the ASD NIM container.
   * - ``ASD_LOG_LEVEL``
     - str
     - ``INFO``
     - Logging level.

LipSync NIM
-----------

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``LIPSYNC_IMAGE``
     - str
     - (set in ``configs/*.env``)
     - LipSync NIM container image reference used by
       ``docker-compose.yml``.
   * - ``LIPSYNC_NIM_GRPC_API_PORT``
     - int
     - ``50054``
     - gRPC listen port.
   * - ``LIPSYNC_NIM_HTTP_API_PORT``
     - int
     - ``8004``
     - HTTP API port (health endpoint) exposed by the LipSync NIM
       container.
   * - ``LIPSYNC_NIM_TAGS_SELECTOR``
     - str
     - ``language=de``
     - NIM tag selector used to pick the LipSync model variant.
   * - ``LIPSYNC_MODEL_MOUNT_PATH``
     - str
     - ``./volumes/models/lipsync``
     - Host path mounted as the LipSync NIM model cache
       (``/opt/nim/.cache``).
   * - ``LIPSYNC_LOG_LEVEL``
     - str
     - ``INFO``
     - Logging level.
   * - ``LIPSYNC_DEBUG_MODE``
     - int
     - ``0``
     - Set to ``1`` to enable LipSync NIM debug mode.
   * - ``NV_AI4M_LS_INPUT_QUEUE_TIMEOUT_S``
     - int
     - ``5``
     - Seconds LipSync waits for input queue data before timing out.

Secrets
-------

These must be set in a ``.env`` file at the repository root (never
committed to version control).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``LIPSYNC_API_KEY``
     - NGC API key for LipSync NIM. Mapped to ``NGC_API_KEY`` inside the
       LipSync container by ``docker-compose.yml``.
   * - ``ASD_API_KEY``
     - NGC API key for ASD NIM. Mapped to ``NGC_API_KEY`` inside the
       ASD container by ``docker-compose.yml``.
   * - ``ELEVENLABS_API_KEY``
     - ElevenLabs API key (required when ``S2S_SERVICE=EL_DUBBING``).
   * - ``CAMB_API_KEY``
     - CambAI API key (required when ``S2S_SERVICE=CAMB_DUBBING``).
