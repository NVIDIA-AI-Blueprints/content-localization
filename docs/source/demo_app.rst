.. _demo_app:

Demo Web Application
====================

The Content Localization Blueprint includes a reference web-based application built with Next.js that demonstrates how to interact with the Controller service.


Overview
--------

The demo application is a Next.js-based web client that provides:

- **Video Input**: Select a video file from your device or use the bundled sample
- **Real-time Processing**: Live preview of content localization with WebSocket
  streaming and pipeline status updates (uploading, preprocessing, localizing)
- **Language Selection**: Configure source and target languages from dropdown menus,
  dynamically populated based on the active S2S backend
- **Preprocessing**: Optional voice isolation and custom diarization upload
  (enabled via ``REFERENCE_APP_ENABLE_PREPROCESSING``)
- **Output Preview and Download**: View processed results and download localized
  videos

Architecture
------------
.. mermaid::

   flowchart LR
       user[User] --> webUi[WebInterface]
       webUi --> upload[VideoUpload]
       webUi --> wsClient[WebSocketClient]

       subgraph uiServer [UIServer]
           nextServer[NextJsServer]
           wsServer[WebSocketServer]
           grpcClient[GrpcClient]
           preprocessing[Preprocessing]
       end

       wsClient <--> wsServer
       webUi --> nextServer
       nextServer --> wsServer
       wsServer --> preprocessing
       preprocessing --> grpcClient

       grpcClient --> controller[ControllerService]
       controller --> grpcClient
       wsServer --> wsClient

The server-side preprocessing step (voice isolation and diarization) runs
between the WebSocket upload and the gRPC call to the Controller when
``REFERENCE_APP_ENABLE_PREPROCESSING`` is enabled.

Deployment
----------

**Docker Compose**

The easiest way to run the demo application is using Docker Compose profiles:

.. code-block:: bash

   # With ElevenLabs/CambAI
   docker compose --profile demo-app-third-party-s2s \
       --env-file configs/elevenlabs.env \
       --env-file .env \
       up --build

Please select the profile name and environment file to suit your needs.

**Access the application:**

.. code-block:: text

   http://localhost:3000

**Remote Access:**

To access from another machine on the same network:

.. code-block:: text

   http://<server-ip-address>:3000

Using the Application
---------------------

1. **Select Languages**: Choose source and target languages from the dropdowns.
   Available languages are determined by the active S2S backend
   (``S2S_SERVICE``).
2. **Upload Video**:

   - Click "Upload Video" or drag and drop a video file
   - Supported format: MP4 (max 1 GB)

3. **Configure Advanced Settings** *(optional, when preprocessing is enabled)*:

   - **Voice Isolation**: Enabled by default. Isolates speech from background
     audio before localization.
   - **Custom Diarization**: Upload a JSON diarization file to guide speaker
     segmentation.

4. **Run Processing**: Click "Run" to begin content localization.
5. **Monitor Progress**: Watch the pipeline status (Uploading → Preprocessing →
   Localizing) and the real-time video preview.
6. **Download Results**: Once complete, download the localized video.

Standalone Development
----------------------

For local development:

**Prerequisites:**

- Node.js 24.x installed
- Controller service running and accessible
- FFmpeg installed

**Step 1: Install Dependencies**

.. code-block:: bash

   cd client/demos
   npm install

**Step 2: Generate Protobuf Code**

.. code-block:: bash

   npm run generate-ts-protos


**Step 3: Start Development Server**

.. code-block:: bash

   npm run dev

**Step 4: Access Application**

Open your browser and navigate to ``http://localhost:3000``

Accessing From Other Devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To access the demo app from other devices on your network:

.. code-block:: bash
   
   # Then access from other devices:
   http://<your-ip-address>:3000
   # Example: http://192.168.1.50:3000

Code Style
~~~~~~~~~~

The project uses ESLint and Prettier for code quality:

.. code-block:: bash

   # Check code quality
   npm run lint

   # Format code
   npm run format

Debugging
~~~~~~~~~

**Enable Debug Logging:**

.. code-block:: bash

   # In configs/elevenlabs.env
   REFERENCE_APP_LOG_LEVEL=debug


**Check Logs:**

- **Browser**: DevTools Console (client-side logs)
- **Server**: Terminal output (server-side logs)
- **WebSocket**: DevTools Network tab → Filter by "WS" (Or "Socket")

Generating Protobuf Code
~~~~~~~~~~~~~~~~~~~~~~~~~

When ``.proto`` files change, regenerate TypeScript definitions:

.. code-block:: bash

   cd client/demos
   npm run generate-ts-protos

This generates TypeScript code in ``app/generated_protos/`` from the protobuf definitions in the ``protos/`` directory.

Configuration
-------------

Connecting to Remote Controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To connect the demo app to a Controller service running on a different machine:

Update the ``CONTROLLER_SERVER`` environment variable in your docker-compose configuration:

.. code-block:: yaml

   # docker-compose.yml
   services:
     demo-app:
       environment:
         # Replace with your remote server IP and port
         - CONTROLLER_SERVER=192.168.1.100:50056

Common Configurations
~~~~~~~~~~~~~~~~~~~~~

The demo app can be configured through environment variables in
``configs/elevenlabs.env``:

.. code-block:: bash

   # Default input video file (served at /api/inputs/<filename>)
   DEFAULT_INPUT_FILE_NAME=sample_video.mp4

   # Demo app log level (mapped to NEXT_PUBLIC_LOG_LEVEL in Docker)
   REFERENCE_APP_LOG_LEVEL=INFO

   # Default languages
   S2S_DEFAULT_SOURCE_LANGUAGE=auto
   S2S_DEFAULT_TARGET_LANGUAGE=de

   # Enable advanced settings (voice isolation, diarization upload).
   REFERENCE_APP_ENABLE_PREPROCESSING=true

Configure Available Languages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Available languages are determined automatically by the ``S2S_SERVICE``
environment variable (``EL_DUBBING`` or ``CAMB_DUBBING``). The ``/api/configs/general`` endpoint returns the
supported language lists and validated defaults for the active backend.

To modify the language lists themselves, edit
``app/api/socketHandlers/content-localization/config.ts``.
