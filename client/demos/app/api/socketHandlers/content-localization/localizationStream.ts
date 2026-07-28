/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import path from "path";
import fs from "fs";
import * as grpc from "@grpc/grpc-js";
import { spawn } from "child_process";
import { nanoid } from "nanoid";
import logger from "../../../utils/logger";
import {
  ContentLocalizationControllerClient,
  ContentLocalizationConfig,
  ContentLocalizationRequest,
  ContentLocalizationResponse,
} from "../../../generated_protos/nvidia/ai4m/controller/v1/controller";
import { AudioCodec } from "../../../generated_protos/nvidia/ai4m/audio/v1/audio";
import {
  ActiveSpeakerDetectionConfig,
  AudioSourceConfig,
  type AudioDiarizationInfo,
} from "../../../generated_protos/nvidia/ai4m/activespeakerdetection/v1/activespeakerdetection";
import { LipsyncConfig } from "../../../generated_protos/nvidia/ai4m/lipsync/v1/lipsync";
import { clientCallHandler } from "../../utils/protoHelper";
import { remuxVideo } from "../../utils/muxDemux";
import { getFileExtension } from "../../../utils/codecConfig";
import { getStreamingArgs } from "../../utils/ffmpegConfig";
import type { CodecId } from "../../../utils/codecConfig";
import { getOutputDir, getServerAddress } from "./config";

const DEFAULT_ASD_SPEAKER_DETECTION_THRESHOLD = 0.5986;
const DEFAULT_LIPSYNC_BITRATE_MBPS = 20;
const DEFAULT_LIPSYNC_IDR_INTERVAL = 8;

export interface LocalizationStreamCallbacks {
  onData: (audio: Buffer, video: Buffer) => void;
  onEnd: () => void;
  cancelProcessing: () => void;
  streamId: string;
}

/**
 * Queues and sends audio/video chunks to the controller over gRPC.
 *
 * Enforces the required message order (per proto and client/controller/app.py):
 * 0. Controller config: one message with controller_config (bypass_asd, etc.).
 * 1. Service configs: one optional asd_config and one lipsync_config.
 * 2. Diarization: zero or more messages with only diarization_info.
 * 3. S2S config: one message with only s2s_config.
 * 4. Data: separate messages with only video_file_data or only audio_data.
 *
 * Chunks are queued via a promise chain so writes never interleave.
 */
class GrpcChunkStreamer {
  private queue: Promise<void> | null = null;
  private ended = false;
  private diarizationSent = false;
  private serviceConfigSent = false;
  private s2sConfigSent = false;
  private controllerConfigSent = false;

  constructor(
    private call: grpc.ClientDuplexStream<ContentLocalizationRequest, ContentLocalizationResponse>,
    private streamId: string,
    private diarizationChunks: AudioDiarizationInfo[],
    private sourceLanguage: string | undefined,
    private targetLanguage: string | undefined,
    private bypassAsd: boolean,
  ) {}

  /** Queue an audio/video chunk pair for sending. */
  public append(audio: Buffer, video: Buffer): void {
    if (this.ended) return;
    if (this.queue) {
      this.queue = this.queue.then(() => this.send(audio, video));
    } else {
      this.queue = this.send(audio, video);
    }
  }

  /** Drain the queue, then close the gRPC stream. */
  public stop(): void {
    if (this.queue) {
      this.queue.then(() => this.end());
    } else {
      this.end();
    }
  }

  private end(): void {
    this.ended = true;
    this.call.end();
  }

  private async send(audio: Buffer, video: Buffer): Promise<void> {
    this.sendControllerConfig();
    this.sendServiceConfigs();
    this.sendDiarization();
    this.sendS2sConfig();
    this.sendData(audio, video);
  }

  /** Step 0: send controller_config (once, before everything else). */
  private sendControllerConfig(): void {
    if (this.controllerConfigSent) return;
    this.call.write(
      ContentLocalizationRequest.fromPartial({
        controller_config: ContentLocalizationConfig.fromPartial({
          bypass_asd: this.bypassAsd,
          // The demo always streams preprocessed WAV; declaring the codec
          // here keeps the controller off its assume-WAV fallback path.
          input_audio_config: {
            encoding: AudioCodec.AUDIO_CODEC_WAV,
          },
        }),
        request_id: this.streamId,
      }),
    );
    this.controllerConfigSent = true;
    logger.debug("[ContentLocalization] Controller config sent", {
      bypass_asd: this.bypassAsd,
    });
  }

  /** Step 1: send downstream service configs (once, before data). */
  private sendServiceConfigs(): void {
    if (this.serviceConfigSent) return;
    if (!this.bypassAsd) {
      this.call.write(
        ContentLocalizationRequest.fromPartial({
          asd_config: ActiveSpeakerDetectionConfig.fromPartial({
            input_audio_config: {
              encoding: AudioCodec.AUDIO_CODEC_WAV,
            },
            audio_source_config: AudioSourceConfig.AUDIO_SOURCE_CONFIG_SEPARATE_STREAM,
            speaker_detection_threshold: DEFAULT_ASD_SPEAKER_DETECTION_THRESHOLD,
          }),
          request_id: this.streamId,
        }),
      );
    }
    this.call.write(
      ContentLocalizationRequest.fromPartial({
        lipsync_config: LipsyncConfig.fromPartial({
          input_audio_codec: AudioCodec.AUDIO_CODEC_MP3,
          output_video_encoding: {
            lossy: {
              bitrate_mbps: DEFAULT_LIPSYNC_BITRATE_MBPS,
              idr_interval: DEFAULT_LIPSYNC_IDR_INTERVAL,
            },
          },
        }),
        request_id: this.streamId,
      }),
    );
    this.serviceConfigSent = true;
    logger.debug("[ContentLocalization] Service configs sent to controller", {
      asd_config: !this.bypassAsd,
      lipsync_config: true,
    });
  }

  /** Step 2: send diarization chunks (once, before s2s_config). */
  private sendDiarization(): void {
    if (this.diarizationSent || this.diarizationChunks.length === 0) return;
    for (const chunk of this.diarizationChunks) {
      this.call.write(
        ContentLocalizationRequest.fromPartial({
          diarization_info: chunk,
          request_id: this.streamId,
        }),
      );
    }
    this.diarizationSent = true;
    logger.debug("[ContentLocalization] Diarization sent to controller", {
      count: this.diarizationChunks.length,
    });
  }

  /** Step 3: send s2s_config (once, after service configs and diarization). */
  private sendS2sConfig(): void {
    if (this.s2sConfigSent) return;
    this.call.write(
      ContentLocalizationRequest.fromPartial({
        s2s_config: {
          source_language: this.sourceLanguage || process.env.DEFAULT_SOURCE_LANGUAGE || "auto",
          target_language: this.targetLanguage || process.env.DEFAULT_TARGET_LANGUAGE || "de",
          voice_name: process.env.VOICE_NAME,
        },
        request_id: this.streamId,
      }),
    );
    this.s2sConfigSent = true;
    logger.debug("[ContentLocalization] Localization request (s2s_config) sent to controller");
  }

  /** Step 4: send video and audio as separate messages. */
  private sendData(audio: Buffer, video: Buffer): void {
    if (video.length > 0) {
      this.call.write(
        ContentLocalizationRequest.fromPartial({
          video_file_data: video,
          request_id: this.streamId,
        }),
      );
    }
    if (audio.length > 0) {
      this.call.write(
        ContentLocalizationRequest.fromPartial({
          audio_data: audio,
          request_id: this.streamId,
        }),
      );
    }
  }
}

/** Wait for a write stream to finish flushing. */
async function drainWriteStream(stream: fs.WriteStream): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    stream.on("finish", resolve);
    stream.on("error", reject);
    stream.end();
  });
}

/**
 * Finalize the output video: flush the file stream, optionally remux for
 * fallback codec, and return the URLs for the client.
 */
async function finalizeOutput(opts: {
  streamId: string;
  outputDir: string;
  outputVideoFilename: string;
  outputVideoFilePath: string;
  outputVideoFileStream: fs.WriteStream;
  downloadCodecId: CodecId;
}): Promise<{ outputVideo: string; fallbackOutputVideo: string }> {
  await drainWriteStream(opts.outputVideoFileStream);

  logger.info("[ContentLocalization] Output video written", {
    streamId: opts.streamId,
    path: opts.outputVideoFilePath,
    bytes: opts.outputVideoFileStream.bytesWritten,
  });

  let fallbackOutputFileName = opts.outputVideoFilename;

  if (opts.downloadCodecId === "fallback") {
    fallbackOutputFileName = `${opts.streamId}_fallback.${getFileExtension(opts.downloadCodecId)}`;
    const fallbackOutputPath = path.join(opts.outputDir, fallbackOutputFileName);
    try {
      await remuxVideo(opts.outputVideoFilePath, fallbackOutputPath, opts.downloadCodecId);
    } catch (err) {
      logger.error("[ContentLocalization] Remux failed, using original", { err });
      fallbackOutputFileName = opts.outputVideoFilename;
    }
  }

  return {
    outputVideo: `/api/outputs/${opts.outputVideoFilename}`,
    fallbackOutputVideo: `/api/outputs/${fallbackOutputFileName}`,
  };
}

/**
 * Creates the content-localization pipeline: gRPC stream to controller + ffmpeg
 * for re-encoding. Returns callbacks to push chunks and to stop the stream.
 */
export function createLocalizationStream(
  sendData: (payload: string) => void,
  options: {
    sourceLanguage?: string;
    targetLanguage?: string;
    streamingCodecId?: CodecId;
    downloadCodecId?: CodecId;
    diarizationChunks?: AudioDiarizationInfo[];
    bypassAsd?: boolean;
  } = {},
): LocalizationStreamCallbacks {
  const {
    sourceLanguage,
    targetLanguage,
    streamingCodecId = "default",
    downloadCodecId = "default",
    diarizationChunks = [],
    bypassAsd = false,
  } = options;

  const serverAddress = getServerAddress();
  const outputDir = getOutputDir();
  const streamId = nanoid(36);

  // gRPC connection to the controller
  const controllerClient = new ContentLocalizationControllerClient(serverAddress, grpc.credentials.createInsecure());
  const call = controllerClient.streamContentLocalization();
  logger.info("[ContentLocalization] Localization stream started", { streamId, serverAddress });

  // Output video file + ffmpeg for real-time streaming preview
  const outputVideoFilename = `${streamId}.mp4`;
  const outputVideoFilePath = path.join(outputDir, outputVideoFilename);
  const outputVideoFileStream = fs.createWriteStream(outputVideoFilePath);
  const ffmpegCommand = spawn("ffmpeg", getStreamingArgs(streamingCodecId));
  let processingEnded = false;

  // Drain stderr to prevent ffmpeg from blocking if the pipe buffer fills
  ffmpegCommand.stderr?.resume();

  ffmpegCommand.stdout.on("data", (data: Buffer) => {
    if (processingEnded) return;
    sendData(
      JSON.stringify({
        type: "video_chunk",
        data: { streamId, chunk: data.toString("base64") },
      }),
    );
  });

  // Upstream: queue chunks to the controller in the required order
  const streamer = new GrpcChunkStreamer(call, streamId, diarizationChunks, sourceLanguage, targetLanguage, bypassAsd);

  let errorOccurred = false;

  // Downstream: handle responses from the controller
  clientCallHandler(call, {
    onData: (response: ContentLocalizationResponse) => {
      if (response.keepalive) return;
      if (response.video_file_data && response.video_file_data.length > 0) {
        const data = Buffer.from(response.video_file_data);
        outputVideoFileStream.write(data);
        ffmpegCommand.stdin.write(data);
      }
    },
    onEnd: async () => {
      ffmpegCommand.stdin.end();
      processingEnded = true;

      if (errorOccurred) {
        logger.error("[ContentLocalization] Skipping final result due to earlier error");
        return;
      }

      const urls = await finalizeOutput({
        streamId,
        outputDir,
        outputVideoFilename,
        outputVideoFilePath,
        outputVideoFileStream,
        downloadCodecId,
      });

      sendData(JSON.stringify({ type: "translate", data: urls }));
      logger.info("[ContentLocalization] Sent result to client", urls);
    },
    onError: (error: Error) => {
      logger.error("[ContentLocalization] Controller error", {
        error: error.message,
        streamId,
      });
      ffmpegCommand.kill();
      errorOccurred = true;
      streamer.stop();
      sendData(JSON.stringify({ type: "translate", data: { error: error.message } }));
    },
  });

  const cancelProcessing = () => {
    logger.info("[ContentLocalization] Cancelling localization stream", { streamId });
    try {
      call.cancel();
      controllerClient.close();
      ffmpegCommand.kill();
    } catch (err) {
      logger.error("[ContentLocalization] Error during cancel", { err });
    }
  };

  return {
    onData: (audio: Buffer, video: Buffer) => streamer.append(audio, video),
    onEnd: () => {
      logger.info("[ContentLocalization] All input chunks received, ending stream", { streamId });
      streamer.stop();
    },
    cancelProcessing,
    streamId,
  };
}
