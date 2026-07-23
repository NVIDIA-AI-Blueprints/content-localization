/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import fs from "fs";
import path from "path";
import logger from "../../utils/logger";
import { isolateAudio as isolateAudioElevenLabs } from "./elevenlabs";
import { isolateAudio as isolateAudioCamb } from "./cambai";
import { saveElevenLabsDiarization, saveCambAiDiarization } from "./elevenLabsDiarization";

const S2S_SERVICE = process.env.S2S_SERVICE;

export interface ProcessedAudio {
  audioFilePath: string;
  /** Set when diarization succeeded; undefined or null when skipped or unavailable. */
  diarizationFilePath?: string | null;
  /** Non-fatal warning from a preprocessing step (e.g. voice isolation or diarization failed). */
  warning?: string;
}

export interface AudioProcessingParams {
  streamId: string;
  audioFilePath: string;
  voiceIsolation: boolean;
  customDiarizationFile?: Buffer | null;
  audioOutputDir: string;
  diarizationOutputDir: string;
  sourceLanguage?: string;
}

/**
 * Whether preprocessing (voice isolation, diarization) is available.
 * Override via REFERENCE_APP_ENABLE_PREPROCESSING.
 */
export function getEnablePreprocessingFromEnv(): boolean {
  return process.env.REFERENCE_APP_ENABLE_PREPROCESSING === "true";
}
/**
 * Validate that the diarization file is valid JSON
 * @param fileData - The file data as Buffer
 * @returns true if valid JSON, false otherwise
 */
export function validateDiarizationFile(fileData: Buffer): boolean {
  try {
    const jsonString = fileData.toString("utf-8");
    JSON.parse(jsonString);
    return true;
  } catch (error) {
    logger.error("Invalid JSON in diarization file:", error);
    return false;
  }
}

/**
 * Save custom diarization file uploaded by user
 * @param streamId - Unique stream identifier
 * @param fileData - The diarization file data as Buffer
 * @param outputDir - Directory where the file should be saved
 * @returns Promise resolving to the saved file path
 */
export async function saveCustomDiarizationFile(
  streamId: string,
  fileData: Buffer,
  outputDir: string,
): Promise<string> {
  // Validate JSON
  if (!validateDiarizationFile(fileData)) {
    throw new Error("Invalid JSON in diarization file");
  }

  // Ensure output directory exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const diarizationFilePath = path.join(outputDir, `${streamId}.json`);
  await fs.promises.writeFile(diarizationFilePath, fileData, "utf-8");
  logger.info(`Custom diarization file saved to: ${diarizationFilePath}`);
  return diarizationFilePath;
}

/**
 * Process audio with advanced settings (voice isolation and/or diarization)
 * Runs voice isolation and diarization in parallel when both are needed
 * @param params - Audio processing parameters
 * @returns Promise resolving to processed audio information
 */
export async function processAudioWithAdvancedSettings(params: AudioProcessingParams): Promise<ProcessedAudio> {
  const {
    streamId,
    audioFilePath,
    voiceIsolation,
    customDiarizationFile,
    audioOutputDir,
    diarizationOutputDir,
    sourceLanguage,
  } = params;

  logger.info(`Processing audio with advanced settings for stream: ${streamId}`, {
    voiceIsolation,
    hasCustomDiarization: !!customDiarizationFile,
  });

  // Ensure output directories exist
  if (!fs.existsSync(audioOutputDir)) {
    fs.mkdirSync(audioOutputDir, { recursive: true });
  }
  if (!fs.existsSync(diarizationOutputDir)) {
    fs.mkdirSync(diarizationOutputDir, { recursive: true });
  }

  let diarizationPromise: Promise<string | null>;
  let isolationPromise: Promise<string> | null = null;
  let warning: string | undefined;

  // If custom diarization file is provided, save it (skip automatic diarization)
  if (customDiarizationFile) {
    logger.info("Saving custom diarization file");
    diarizationPromise = saveCustomDiarizationFile(streamId, customDiarizationFile, diarizationOutputDir);
  } else if (S2S_SERVICE === "CAMB_DUBBING") {
    // Camb AI diarization: parse sourceLanguage as numeric language ID
    logger.info("Performing automatic diarization via Camb AI");
    const languageId = sourceLanguage ? parseInt(sourceLanguage, 10) || 1 : 1;
    diarizationPromise = saveCambAiDiarization(streamId, audioFilePath, diarizationOutputDir, languageId);
  } else {
    // Default: ElevenLabs diarization (in parallel with voice isolation if needed)
    logger.info("Performing automatic diarization via ElevenLabs");
    diarizationPromise = saveElevenLabsDiarization(streamId, audioFilePath, diarizationOutputDir);
  }

  // If voice isolation is enabled, perform it (in parallel with diarization)
  if (voiceIsolation) {
    const isolatedOutputPath = path.join(audioOutputDir, `${streamId}_isolated.wav`);
    if (S2S_SERVICE === "CAMB_DUBBING") {
      logger.info("Performing voice isolation via Camb AI");
      isolationPromise = isolateAudioCamb(audioFilePath, isolatedOutputPath);
    } else {
      logger.info("Performing voice isolation via ElevenLabs");
      isolationPromise = isolateAudioElevenLabs(audioFilePath, isolatedOutputPath);
    }
  }

  // Run both in parallel; each settles independently so one failure doesn't affect the other
  const [diarizationResult, isolationResult] = await Promise.allSettled([diarizationPromise, isolationPromise]);

  const diarizationFailed = diarizationResult.status === "rejected" || diarizationResult.value === null;
  // Only treat isolation as failed when it was actually requested

  const isolationFailed = voiceIsolation && (isolationResult.status === "rejected" || isolationResult.value === null);

  if (diarizationFailed) {
    logger.error("Diarization failed. Active speaker detection results may be less accurate.", {
      error: diarizationResult.status === "rejected" ? diarizationResult.reason : "Unknown error",
    });
  }
  if (isolationFailed) {
    logger.error("Voice isolation failed. Active speaker detection will run on original audio.", {
      error: isolationResult.status === "rejected" ? isolationResult.reason : "Unknown error",
    });
  }

  // Single consolidated warning when both fail, otherwise individual warning
  if (diarizationFailed && isolationFailed) {
    warning = "Audio pre-processing could not be completed. The results may be less accurate.";
  } else if (diarizationFailed) {
    warning = "Diarization failed. Active speaker detection results may be less accurate.";
  } else if (isolationFailed) {
    warning =
      "Audio pre-processing partially failed. Active speaker detection will run on the original audio and results may be less accurate.";
  }

  const diarizationFilePath: string | null | undefined = diarizationFailed ? null : diarizationResult.value;

  let isolatedAudioPath: string | undefined;
  if (isolationResult.status === "fulfilled" && isolationResult.value) {
    isolatedAudioPath = isolationResult.value;
  }

  // Return the isolated audio path if voice isolation succeeded, otherwise return original
  const finalAudioPath = isolatedAudioPath || audioFilePath;

  logger.info(`Audio processing completed for stream: ${streamId}`, {
    finalAudioPath,
    diarizationFilePath,
    usedIsolatedAudio: !!isolatedAudioPath,
    warning,
  });

  return {
    audioFilePath: finalAudioPath,
    diarizationFilePath,
    warning,
  };
}
