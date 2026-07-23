# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller Service for Content Localization.

This module implements ``ControllerService``, the main service class that
orchestrates the content localization pipeline. It coordinates between
client applications and downstream AI services to provide end-to-end
content localization capabilities.

Architecture Overview
=====================

The Controller Service acts as the central orchestrator in a microservices
architecture, coordinating three main AI services:

1. **Speech-to-Speech (S2S) Service**: Handles audio translation and synthesis
2. **Active Speaker Detection (ASD) Service**: Identifies speaking faces in video
3. **LipSync Service**: Synchronizes lip movements with translated audio

``ControllerService`` manages:

- Deserializer-based request distribution
- Multi-threaded client processing (via ``controller_service.pipeline``)
- Response streaming
- Error handling and recovery

The gRPC boundary lives in ``controller_service.servicer`` and the
per-request thread-orchestration helpers live in
``controller_service.pipeline``.

Request Flow
============

See :mod:`controller_service.pipeline` for the request-flow diagram.

Configuration Management
========================

- S2S output audio format derived from ``S2S_SERVICE`` env var
  (MP3 for ElevenLabs and CambAI)
- ``is_speaker_info_provided`` auto-set based on ASD availability
- ASD service optional (bypass_asd per-request)
- gRPC message size limits
- Service endpoint configuration

Usage
=====

The Controller Service is typically deployed as a Docker container and
exposes a gRPC interface for client applications:

.. code-block:: python

    # Client usage example
    import grpc
    from nvidia.ai4m.controller.v1.controller_pb2_grpc import ContentLocalizationControllerStub

    channel = grpc.insecure_channel("localhost:50056")
    stub = ContentLocalizationControllerStub(channel)

    # Stream requests to the service
    responses = stub.StreamContentLocalization(request_iterator)
    for response in responses:
        # Process video data
        if response.HasField("video_file_data"):
            # Handle video data
            pass

Configuration
=============

The service can be configured through environment variables:
- ```CONTROLLER_GRPC_API_PORT```: gRPC service port
- ```S2S_SERVER```: S2S service endpoint (optional, bypass_s2s when not provided)
- ```ASD_SERVER```: ASD service endpoint (optional, bypass_asd when not provided)
- ```LIPSYNC_SERVER```: LipSync service endpoint
- ```S2S_SERVICE```: S2S service type (e.g., "EL_DUBBING" for ElevenLabs)
"""

import argparse
import os
import threading
from collections.abc import Iterator

import grpc
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_WAV
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationRequest
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationResponse
from nvidia.ai4m.controller.v1.controller_pb2_grpc import (
    add_ContentLocalizationControllerServicer_to_server,
)
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from common.base_utils import GRPCServiceBase
from common.base_utils import logger
from common.buffers import Buffer
from common.errors import PipelineInputError
from common.errors import PipelinePreconditionError
from common.nims import ActiveSpeakerDetectionHandle
from common.nims import LipsyncHandle
from common.nims import SpeechToSpeechHandle
from common.service import message_size_channel_options
from controller_service import pipeline
from controller_service.config import _PipelineConfig
from controller_service.config import _RequestServices
from controller_service.deserializer import ContentLocalizationDeserializer
from controller_service.helpers import _FORMAT_TO_CODEC
from controller_service.helpers import _S2S_OUTPUT_FORMAT
from controller_service.helpers import _audio_codec_to_format_string
from controller_service.helpers import _extract_config
from controller_service.servicer import ControllerServiceServicer

_SSL_MODES = ("DISABLED", "TLS", "MTLS")


def _ssl_mode_env_default(env_var: str, fallback: str | None = None) -> str | None:
    """Read and validate an SSL-mode default from an environment variable.

    argparse only enforces ``choices`` on command-line values, not on
    defaults, so an invalid environment value would otherwise flow
    silently into the channel-security configuration.

    Args:
        env_var (str): Name of the environment variable to read.
        fallback (str | None): Value returned when the variable is unset
            or empty. Defaults to ``None``.

    Returns:
        str | None: The validated mode, or ``fallback`` when unset.

    Raises:
        ValueError: If the variable is set to something other than
            ``DISABLED``, ``TLS``, or ``MTLS``.

    Examples:
        >>> _ssl_mode_env_default(env_var="UNSET_VAR", fallback="DISABLED")
        'DISABLED'
    """
    value = os.getenv(env_var)
    if not value:
        return fallback
    if value not in _SSL_MODES:
        raise ValueError(f"{env_var} must be one of {', '.join(_SSL_MODES)}, got {value!r}")
    return value


class ControllerService(GRPCServiceBase):
    """Main Controller Service for orchestrating the content localization pipeline.

    The ControllerService is the central orchestrator that coordinates between client
    applications and downstream AI services to provide end-to-end content localization
    capabilities. It uses a multi-threaded deserializer + buffer + client thread
    architecture.

    Responsibilities
    ================

    - **Service Orchestration**: Coordinates communication with S2S, ASD, and LipSync services
    - **Request Distribution**: Uses ContentLocalizationDeserializer to distribute
      requests to buffers
    - **Multi-threaded Processing**: Manages concurrent client threads for each service
    - **Response Streaming**: Streams processed video data back to clients
    - **Error Handling**: Provides comprehensive error handling and recovery

    Multi-Threaded Architecture
    ===========================

    The service uses the following architecture:

    Deserializer + Buffer + Client Threads
    ---------------------------------------
    - **ContentLocalizationDeserializer**: Background thread consumes gRPC request stream
    - **Typed Buffers**: Distribute requests to appropriate services
      (audio_buffer, video_buffer, translated_audio_buffer, controller_config_buffer, etc.)
    - **Client Threads**: Concurrent processing for each service
      - S2S Client Thread: Processes audio translation (skipped in bypass-S2S mode)
      - ASD Client Thread: Detects active speakers (skipped when bypass_asd)
      - LipSync Client Thread: Generates lip-synced video
    - **Bypass-S2S Mode**: When ``controller_config.bypass_s2s`` is True,
      S2S is skipped and ``translated_audio_buffer`` feeds LipSync directly
    - **Main Thread**: Yields ContentLocalizationResponse to client

    Key Components
    ==============

    Downstream Services
    -------------------
    - ```lipsync_server```: LipSync service for video processing
    - ```s2s_server```: Speech-to-Speech service for audio translation
    - ```asd_server```: Active Speaker Detection service (optional, bypass_asd when None)

    Configuration
    -------------
    - ```s2s_output_audio_format```: Derived from ``S2S_SERVICE`` env var.
      Fixed to ``"MP3"`` for both ElevenLabs and CambAI. Used as LipSync
      input codec only when S2S is active.
    - ```message_size```: Maximum gRPC message size

    Request Flow
    ============

    See :mod:`controller_service.pipeline` for the request-flow diagram.

    Error Handling
    ==============

    The service provides comprehensive error handling:
    - gRPC communication errors with downstream services
    - Request processing errors in client threads
    - Service unavailability handling
    - Proper error propagation to clients
    - Graceful degradation when ASD is bypassed (bypass_asd)

    Thread Safety
    =============

    The service is designed to be thread-safe:
    - Deserializer runs on dedicated background thread
    - Each service has dedicated client thread(s)
    - Buffer-based communication ensures thread isolation
    - Proper resource cleanup and thread management in finally blocks

    Usage
    =====

    The ControllerService is typically instantiated with downstream service
    connections:

    .. code-block:: python

        service = ControllerService(
            lipsync_server=lipsync_server,
            s2s_server=s2s_server,  # Optional - None for bypass-S2S-only
            asd_server=asd_server,  # Optional — bypass_asd when None
        )
    """

    def __init__(
        self,
        lipsync_server: LipsyncHandle,
        s2s_server: SpeechToSpeechHandle | None = None,
        asd_server: ActiveSpeakerDetectionHandle | None = None,
        message_size: int = 1024 * 1024 * 4,
    ) -> None:
        """Initialize the controller service.

        Args:
            lipsync_server (LipsyncHandle): Handle to the LipSync service.
            s2s_server (SpeechToSpeechHandle | None): Handle to the S2S service.
                ``None`` when running in bypass-S2S-only mode (all
                requests must set ``bypass_s2s=True``).
            asd_server (ActiveSpeakerDetectionHandle | None): Handle to the
                ASD service. If None, ASD is disabled.
            message_size (int): The maximum message size in bytes.
                Defaults to 1024 * 1024 * 4.
        """
        super().__init__(message_size=message_size)

        self.lipsync_server = lipsync_server
        self.s2s_server = s2s_server
        self.asd_server = asd_server

        # S2S output format is only needed when S2S is configured.
        if self.s2s_server is not None:
            s2s_service = os.environ.get("S2S_SERVICE", "EL_DUBBING")
            if s2s_service not in _S2S_OUTPUT_FORMAT:
                raise ValueError(
                    f"Unknown S2S_SERVICE={s2s_service!r}. "
                    f"Supported values: {list(_S2S_OUTPUT_FORMAT)}"
                )
            self.s2s_output_audio_format: str = _S2S_OUTPUT_FORMAT[s2s_service]
            logger.info(
                f"S2S backend={s2s_service}, s2s_output_audio_format={self.s2s_output_audio_format}"
            )
        else:
            self.s2s_output_audio_format: str = ""
            logger.info("S2S service not configured — only bypass-S2S requests supported")

        logger.debug("Controller Service initialized (deserializer + buffer + client threads)")

        if self.asd_server is None:
            logger.debug("ASD service disabled - LipSync will use internal face detection")

    @staticmethod
    def argsfactory(
        parser: argparse.ArgumentParser | None = None,
    ) -> argparse.ArgumentParser:
        """Parser for command line arguments.

        Extends the vendored ``GRPCServiceBase.argsfactory`` surface (which
        already provides the server-side ``--use-ssl``/``--ssl_server_*``
        options) with downstream service URIs and the client-side
        ``--ssl-mode``/``--ssl-key``/``--ssl-cert``/``--ssl-root-cert``
        options for the NIM channels.

        Args:
            parser (argparse.ArgumentParser | None): Optional existing parser to extend.

        Returns:
            argparse.ArgumentParser: Unparsed command line arguments

        Examples:
            >>> parser = ControllerService.argsfactory()
            >>> args = parser.parse_args(["--lipsync-server", "lipsync:50054"])
            >>> args.ssl_mode
            'DISABLED'
        """
        if parser is None:
            parser = argparse.ArgumentParser(description="Controller Service")

        parser = GRPCServiceBase.argsfactory(parser)

        # The controller streams media chunks of up to 1 MiB, so the gRPC
        # service base's 64 KiB --message-size default is too small for the
        # downstream NIM channels. Default to 4 MiB, matching gRPC's own
        # receive limit.
        parser.set_defaults(message_size=1024 * 1024 * 4)

        parser.add_argument(
            "--s2s-server",
            type=str,
            required=False,
            help="S2S service URI (host:port). Not required when running in bypass-S2S-only mode.",
        )
        parser.add_argument(
            "--asd-server",
            type=str,
            required=False,
            help="ASD NIM service URI (host:port). When omitted, only "
            "bypass_asd=True requests are supported.",
        )
        parser.add_argument(
            "--lipsync-server",
            type=str,
            required=True,
            help="LipSync NIM service URI (host:port)",
        )

        # Client-side channel security for the downstream S2S/ASD/LipSync
        # connections. Mirrors the client-app --ssl-mode surface and reuses
        # common.tls.create_channel_credentials. Defaults come from env vars
        # so compose deployments can enable TLS without editing entrypoints.
        parser.add_argument(
            "--ssl-mode",
            type=str,
            choices=list(_SSL_MODES),
            default=_ssl_mode_env_default(
                env_var="CONTROLLER_NIM_SSL_MODE",
                fallback="DISABLED",
            ),
            help="Channel security for downstream NIM connections. Can also be "
            "set via CONTROLLER_NIM_SSL_MODE. Default is DISABLED (plaintext).",
        )
        parser.add_argument(
            "--ssl-key",
            type=str,
            default=os.getenv("CONTROLLER_NIM_SSL_KEY") or None,
            help="Path to the client private key PEM for MTLS downstream "
            "channels. Can also be set via CONTROLLER_NIM_SSL_KEY.",
        )
        parser.add_argument(
            "--ssl-cert",
            type=str,
            default=os.getenv("CONTROLLER_NIM_SSL_CERT") or None,
            help="Path to the client certificate chain PEM for MTLS downstream "
            "channels. Can also be set via CONTROLLER_NIM_SSL_CERT.",
        )
        parser.add_argument(
            "--ssl-root-cert",
            type=str,
            default=os.getenv("CONTROLLER_NIM_SSL_ROOT_CERT") or None,
            help="Path to the root certificate PEM used to verify downstream "
            "NIM servers (TLS and MTLS). Can also be set via "
            "CONTROLLER_NIM_SSL_ROOT_CERT.",
        )

        # Per-service overrides of --ssl-mode, so hops with different TLS
        # capability can coexist (e.g. TLS-terminated services next to
        # hops that are still plaintext).
        # PEM material stays shared via the global --ssl-key/--ssl-cert/
        # --ssl-root-cert options.
        parser.add_argument(
            "--s2s-ssl-mode",
            type=str,
            choices=list(_SSL_MODES),
            default=_ssl_mode_env_default(env_var="CONTROLLER_S2S_SSL_MODE"),
            help="Channel security for the S2S hop; overrides --ssl-mode. "
            "Can also be set via CONTROLLER_S2S_SSL_MODE. Defaults to the "
            "global --ssl-mode.",
        )
        parser.add_argument(
            "--asd-ssl-mode",
            type=str,
            choices=list(_SSL_MODES),
            default=_ssl_mode_env_default(env_var="CONTROLLER_ASD_SSL_MODE"),
            help="Channel security for the ASD hop; overrides --ssl-mode. "
            "Can also be set via CONTROLLER_ASD_SSL_MODE. Defaults to the "
            "global --ssl-mode.",
        )
        parser.add_argument(
            "--lipsync-ssl-mode",
            type=str,
            choices=list(_SSL_MODES),
            default=_ssl_mode_env_default(env_var="CONTROLLER_LIPSYNC_SSL_MODE"),
            help="Channel security for the LipSync hop; overrides --ssl-mode. "
            "Can also be set via CONTROLLER_LIPSYNC_SSL_MODE. Defaults to "
            "the global --ssl-mode.",
        )

        return parser

    def _check_services_health(
        self,
        services: "_RequestServices",
        bypass_s2s: bool = False,
        bypass_asd: bool = False,
    ) -> None:
        """Check health and preconditions for configured services.

        LipSync is always required. S2S and ASD are checked only
        when not bypassed.

        Args:
            services (_RequestServices): Per-request service handles.
            bypass_s2s (bool): Skip S2S health check.
            bypass_asd (bool): Skip ASD health check.

        Returns:
            None: When all checks pass.

        Raises:
            PipelinePreconditionError: If a required server is not configured.
            ConnectionError: If a required server health check fails.

        Examples:
            >>> svc = ControllerService(...)  # doctest: +SKIP
            >>> svc._check_services_health(
            ...     services=services,
            ...     bypass_s2s=True,
            ...     bypass_asd=False,
            ... )
        """
        services.lipsync_server.is_healthy()
        if not bypass_s2s:
            if services.s2s_server is None:
                msg = (
                    "S2S server is not configured but bypass_s2s is "
                    "not set. Provide --s2s-server or set "
                    "bypass_s2s=True in ContentLocalizationConfig."
                )
                logger.error(msg)
                raise PipelinePreconditionError(msg)
            services.s2s_server.is_healthy()
        if not bypass_asd:
            if services.asd_server is None:
                msg = (
                    "ASD server is not configured but bypass_asd is "
                    "not set. Provide --asd-server or set "
                    "bypass_asd=True in ContentLocalizationConfig."
                )
                logger.error(msg)
                raise PipelinePreconditionError(msg)
            services.asd_server.is_healthy()

    def add_servicer_to_server(self, server: grpc.Server) -> None:
        """Add the controller servicer to the gRPC server."""
        servicer = ControllerServiceServicer(self)
        add_ContentLocalizationControllerServicer_to_server(servicer, server)
        logger.debug("Added controller servicer to gRPC server")

    def _create_request_services(self) -> "_RequestServices":
        """Create per-request channels to the configured downstream services.

        Clones the configured service handles and opens a fresh channel on
        each clone, so this request's channel lifecycle is isolated from
        concurrent requests. Channel options apply the configured
        ``message_size`` to every downstream hop.

        Returns:
            _RequestServices: Connected per-request service handles.

        Raises:
            Exception: Whatever a downstream ``connect()`` raises; any
                channels already opened for this request are closed first.

        Examples:
            >>> services = svc._create_request_services()  # doctest: +SKIP
            >>> services.close()  # doctest: +SKIP
        """
        channel_options = message_size_channel_options(message_size=self.message_size)
        services = _RequestServices(
            lipsync_server=self.lipsync_server.clone(),
            s2s_server=self.s2s_server.clone() if self.s2s_server is not None else None,
            asd_server=self.asd_server.clone() if self.asd_server is not None else None,
        )
        try:
            services.lipsync_server.connect(channel_options=channel_options)
            if services.s2s_server is not None:
                services.s2s_server.connect(channel_options=channel_options)
            if services.asd_server is not None:
                services.asd_server.connect(channel_options=channel_options)
        except Exception:
            # A failed later connect() must not leak the channels opened
            # before it; close() is a no-op on never-connected handles.
            services.close()
            raise
        return services

    def infer(
        self,
        request_iterator: Iterator[ContentLocalizationRequest],
        context: grpc.ServicerContext,
        request_id: str,
    ) -> Iterator[ContentLocalizationResponse]:
        """Main inference method for content localization processing.

        This method orchestrates the complete content localization pipeline, from
        client request ingestion through response streaming. It handles service
        initialization, health checks, and processes requests using the
        multi-threaded deserializer + buffer + client thread architecture.

        Processing Pipeline:
            1. Service Initialization - Creates connections to downstream services
            2. Health Check - Verifies all required services are available
            3. Request Processing - Multi-threaded processing via deserializer
            4. Response Streaming - Streams processed video data back to client
            5. Error Handling - Provides comprehensive error handling and recovery

        Architecture:
            - ContentLocalizationDeserializer: Background thread consumes gRPC stream
            - Typed Buffers: Distribute requests (audio_buffer, video_buffer)
            - Client Threads: Concurrent processing for S2S, ASD, and LipSync
            - Main Thread: Yields ContentLocalizationResponse to client

        Error Handling:
            Exceptions from service creation, health checks, request
            validation, and response processing propagate unchanged to the
            servicer — the pipeline's single abort point — which maps them to
            the most specific gRPC status code. Per-request channels are
            closed on every exit path.

        Args:
                request_iterator: Stream of client requests containing audio/video data
                context: gRPC context for request metadata and cancellation
                request_id: Unique identifier for this request session

        Yields:
                ContentLocalizationResponse: Stream of processed video data

        Raises:
                grpc.RpcError: If there's an error in processing the stream or service communication
        """
        # Exceptions propagate unchanged to the servicer, the pipeline's single
        # abort point, so specific status codes are preserved end to end.
        services = self._create_request_services()
        try:
            logger.debug("Yielding from controller output iterator")
            yield from self._controller_impl(
                request_iterator=request_iterator,
                context=context,
                request_id=request_id,
                services=services,
            )
        finally:
            # Close this request's NIM channels so its connections terminate
            # immediately. Channels are per-request, so closing them cannot
            # affect other in-flight requests.
            services.close()
        logger.info("Controller inference finished")

    def _extract_and_apply_configs(
        self,
        deserializer: ContentLocalizationDeserializer,
    ) -> "_PipelineConfig":
        """Extract client configs from buffers and apply server overrides.

        Reads controller, ASD, and LipSync configs from the
        deserializer's config buffers, then applies server-side
        overrides (``is_speaker_info_provided``, input audio codec).

        Args:
            deserializer (ContentLocalizationDeserializer): Active
                deserializer with populated config buffers.

        Returns:
            _PipelineConfig: Bundled pipeline configuration.

        Examples:
            >>> cfg = svc._extract_and_apply_configs(  # doctest: +SKIP
            ...     deserializer=des,
            ... )
        """
        controller_config = _extract_config(
            deserializer.controller_config_buffer, "controller_config"
        )
        bypass_s2s = controller_config.bypass_s2s if controller_config else False
        bypass_asd = controller_config.bypass_asd if controller_config else False
        # Single predicate for every ASD gate in the pipeline.
        asd_active = not bypass_asd and self.asd_server is not None

        # Only block on ASD config extraction when ASD is active
        asd_config = None
        if asd_active:
            asd_config = _extract_config(deserializer.asd_config_buffer, "asd_config")
        lipsync_config = _extract_config(deserializer.lipsync_config_buffer, "lipsync_config")

        if lipsync_config is None:
            lipsync_config = LipsyncConfig()

        # is_speaker_info_provided is driven by whether ASD runs in this request
        lipsync_config.is_speaker_info_provided = asd_active

        # The client declares its input audio codec on the controller config;
        # for older clients the ASD config's audio encoding is used, and WAV
        # is assumed when neither is present.
        if controller_config is not None and controller_config.HasField("input_audio_config"):
            input_audio_codec = controller_config.input_audio_config.encoding
        elif asd_config is not None:
            input_audio_codec = asd_config.input_audio_config.encoding
        else:
            logger.warning(
                "No input_audio_config on controller_config and no asd_config; "
                "assuming WAV input audio"
            )
            input_audio_codec = AUDIO_CODEC_WAV
        input_audio_format = _audio_codec_to_format_string(input_audio_codec)

        # Override lipsync input codec when S2S is active so the
        # LipSync NIM knows what audio container to expect.
        s2s_output_format: str | None = None
        if not bypass_s2s and self.s2s_server is not None:
            s2s_output_format = self.s2s_output_audio_format

            expected_codec = _FORMAT_TO_CODEC[s2s_output_format]
            if lipsync_config.input_audio_codec != expected_codec:
                logger.warning(
                    f"Client lipsync_config.input_audio_codec "
                    f"({lipsync_config.input_audio_codec}) does not "
                    f"match S2S output format "
                    f"({s2s_output_format}). "
                    f"Overriding to {expected_codec}."
                )
                lipsync_config.input_audio_codec = expected_codec

        return _PipelineConfig(
            bypass_s2s=bypass_s2s,
            bypass_asd=bypass_asd,
            asd_active=asd_active,
            asd_config=asd_config,
            lipsync_config=lipsync_config,
            input_audio_format=input_audio_format,
            s2s_output_format=s2s_output_format,
        )

    def _controller_impl(
        self,
        request_iterator: Iterator[ContentLocalizationRequest],
        context: grpc.ServicerContext,
        request_id: str,
        services: "_RequestServices",
    ) -> Iterator[ContentLocalizationResponse]:
        """Core controller orchestrator for content localization.

        Thin orchestrator that delegates to helper methods and the
        thread-orchestration functions in ``controller_service.pipeline``:

        1. Start deserializer to consume the gRPC request stream
        2. Extract and apply configs
           (``_extract_and_apply_configs``)
        3. Health-check required services
           (``_check_services_health``)
        4. Launch S2S, ASD, LipSync, and unused-input drain threads
        5. Yield responses (``pipeline._yield_responses``)
        6. Validate bypass_s2s / translated-audio pairing
           (``pipeline._validate_translated_audio_usage``)
        7. Clean up threads (``pipeline._cleanup_threads``)

        Both ``bypass_s2s`` and ``bypass_asd`` are per-request
        flags read from ``ContentLocalizationConfig``.

        See :mod:`controller_service.pipeline` for the request-flow diagram.

        Args:
            request_iterator (Iterator[ContentLocalizationRequest]):
                Client request stream.
            context (grpc.ServicerContext): gRPC context.
            request_id (str): Unique request identifier.
            services (_RequestServices): Per-request downstream service
                handles with open channels.

        Yields:
            ContentLocalizationResponse: Processed video data.

        Raises:
            grpc.RpcError: On service communication errors.
        """
        logger.info(f"Service invoked for request id: {request_id}")
        logger.debug("Using Deserializer + Client thread pipeline")

        # --- 1. Deserializer: consume gRPC stream into buffers ---
        deserializer = ContentLocalizationDeserializer(request_iterator)
        deserializer.start(request_id=request_id)
        logger.debug("Deserializer thread started")

        # Threads are registered as they launch so the finally block joins every
        # started thread on all exit paths, including setup failures.
        threads: list[threading.Thread] = []
        try:
            # --- 2. Extract configs and apply server overrides ---
            cfg = self._extract_and_apply_configs(deserializer=deserializer)

            if cfg.bypass_s2s:
                logger.info("Bypass-S2S mode: using translated audio for LipSync")
            else:
                logger.info(
                    f"Audio formats: input={cfg.input_audio_format}, "
                    f"s2s_output={cfg.s2s_output_format or 'unresolved'}"
                )
            if cfg.bypass_asd:
                logger.info("Bypass-ASD mode: LipSync will use internal face detection")

            # --- 3. Health check (validates preconditions too) ---
            self._check_services_health(
                services=services,
                bypass_s2s=cfg.bypass_s2s,
                bypass_asd=cfg.bypass_asd,
            )

            if cfg.asd_active and cfg.asd_config is None:
                raise PipelineInputError(
                    "ASD is active for this request but no asd_config was "
                    "received. Send asd_config at the start of the stream "
                    "or set bypass_asd=True in ContentLocalizationConfig."
                )

            # --- 4. Launch pipeline threads ---
            s2s_output_buffer: Buffer[SpeechToSpeechResponse] = Buffer(num_queues=1)
            s2s_thread = pipeline._start_s2s_thread(
                deserializer=deserializer,
                s2s_server=services.s2s_server,
                s2s_output_buffer=s2s_output_buffer,
                input_audio_format=cfg.input_audio_format,
                bypass_s2s=cfg.bypass_s2s,
                context=context,
                request_id=request_id,
            )
            if s2s_thread is not None:
                threads.append(s2s_thread)
            asd_thread, asd_output_buffer = pipeline._start_asd_thread(
                deserializer=deserializer,
                asd_server=services.asd_server,
                asd_config=cfg.asd_config,
                asd_active=cfg.asd_active,
                context=context,
                request_id=request_id,
            )
            if asd_thread is not None:
                threads.append(asd_thread)
            lipsync_thread, lipsync_output_buffer = pipeline._start_lipsync_thread(
                deserializer=deserializer,
                lipsync_server=services.lipsync_server,
                s2s_output_buffer=s2s_output_buffer,
                asd_output_buffer=asd_output_buffer,
                pipeline_config=cfg,
                context=context,
                request_id=request_id,
            )
            threads.append(lipsync_thread)
            threads.extend(
                pipeline._start_unused_input_drains(
                    deserializer=deserializer,
                    pipeline_config=cfg,
                    request_id=request_id,
                )
            )

            # --- 5. Yield responses to gRPC client ---
            # Responses echo the client-supplied request_id when one was sent.
            # The yield loop rejects forbidden translated audio before any
            # response is emitted (invalid requests must produce no output).
            yield from pipeline._yield_responses(
                lipsync_output_buffer=lipsync_output_buffer,
                request_id=deserializer.client_request_id or request_id,
                deserializer=deserializer,
                bypass_s2s=cfg.bypass_s2s,
            )

            # --- 6. Validate bypass_s2s / translated-audio pairing ---
            pipeline._validate_translated_audio_usage(
                deserializer=deserializer,
                bypass_s2s=cfg.bypass_s2s,
            )
        finally:
            # --- 7. Cleanup ---
            pipeline._cleanup_threads(
                deserializer=deserializer,
                threads=threads,
            )
