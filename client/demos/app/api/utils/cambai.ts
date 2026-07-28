/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Camb AI API client for diarization and audio isolation (voice separation).
 *
 * Diarization flow: POST /transcribe → poll GET /transcribe/{task_id}
 * → GET /transcription-result/{run_id}?word_level_timestamps=true
 *
 * Isolation flow: POST /audio-separation → poll GET /audio-separation/{task_id}
 * → GET /audio-separation-result/{run_id} → download foreground_audio_url
 */

import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import logger from "../../utils/logger";

const CAMB_API_BASE_URL = "https://client.camb.ai/apis";
const CAMB_API_KEY = process.env.CAMB_API_KEY || "";

const POLL_INTERVAL_MS = 10_000;
const MAX_POLL_ATTEMPTS = 120;

/**
 * Submit a transcription request to Camb AI.
 * @param audioFilePath - Path to the audio file to transcribe
 * @param languageId - Camb AI numeric language ID (default: 1 for English)
 * @returns The task ID for polling
 */
async function submitTranscription(audioFilePath: string, languageId: number = 1): Promise<string> {
  const fileBuffer = fs.readFileSync(audioFilePath);
  const fileName = path.basename(audioFilePath);

  const formData = new FormData();
  formData.append("media_file", new Blob([fileBuffer]), fileName);
  formData.append("language", String(languageId));

  const response = await fetch(`${CAMB_API_BASE_URL}/transcribe`, {
    method: "POST",
    headers: { "x-api-key": CAMB_API_KEY },
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Camb AI /transcribe failed (${response.status}): ${text}`);
  }

  const data = await response.json();
  const taskId = data.task_id;
  if (!taskId) {
    throw new Error(`Camb AI /transcribe response missing task_id: ${JSON.stringify(data)}`);
  }
  return String(taskId);
}

/**
 * Poll Camb AI transcription status until SUCCESS.
 * @param taskId - Task ID from submitTranscription
 * @returns The run_id for fetching results
 */
async function waitForTranscription(taskId: string): Promise<number> {
  for (let attempt = 0; attempt < MAX_POLL_ATTEMPTS; attempt++) {
    const response = await fetch(`${CAMB_API_BASE_URL}/transcribe/${taskId}`, {
      headers: { "x-api-key": CAMB_API_KEY },
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Camb AI status check failed (${response.status}): ${text}`);
    }

    const data = await response.json();
    const status = String(data.status ?? "").toUpperCase();

    if (status === "SUCCESS") {
      const runId = data.run_id;
      if (typeof runId !== "number") {
        throw new Error(`Camb AI status missing run_id on SUCCESS: ${JSON.stringify(data)}`);
      }
      return runId;
    }

    if (["ERROR", "TIMEOUT", "PAYMENT_REQUIRED"].includes(status)) {
      throw new Error(`Camb AI transcription failed: status=${status}, message=${data.message}`);
    }

    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }

  throw new Error(`Camb AI transcription timed out after ${MAX_POLL_ATTEMPTS} attempts`);
}

/**
 * Fetch the transcription result with word-level timestamps.
 * @param runId - Run ID from waitForTranscription
 * @returns The transcription result JSON (array of segments)
 */
async function getTranscriptionResult(runId: number): Promise<any> {
  const url = `${CAMB_API_BASE_URL}/transcription-result/${runId}?word_level_timestamps=true`;
  const response = await fetch(url, {
    headers: { "x-api-key": CAMB_API_KEY },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Camb AI /transcription-result failed (${response.status}): ${text}`);
  }

  return response.json();
}

/**
 * Perform Camb AI diarization: submit → poll → fetch result.
 * Returns null if CAMB_API_KEY is not set (graceful skip).
 * @param audioFilePath - Path to the audio file
 * @param streamId - Unique stream identifier for logging
 * @param languageId - Camb AI numeric language ID (default: 1)
 * @returns Transcription result or null if skipped
 */
export async function performCambAiDiarization(
  audioFilePath: string,
  streamId: string,
  languageId: number = 1,
): Promise<any | null> {
  if (!CAMB_API_KEY) {
    logger.warn("CAMB_API_KEY not set. Skipping Camb AI diarization.");
    return null;
  }

  if (!fs.existsSync(audioFilePath)) {
    throw new Error(`Audio file not found for diarization: ${audioFilePath}`);
  }

  logger.info(`Starting Camb AI diarization for: ${audioFilePath} (stream: ${streamId})`);

  try {
    const taskId = await submitTranscription(audioFilePath, languageId);
    logger.info(`Camb AI transcription submitted: taskId=${taskId}`);

    const runId = await waitForTranscription(taskId);
    logger.info(`Camb AI transcription completed: runId=${runId}`);

    const result = await getTranscriptionResult(runId);
    logger.info(`Camb AI diarization completed for stream: ${streamId}`);

    return result;
  } catch (error) {
    logger.error(`Error during Camb AI diarization: ${error}`);
    throw error;
  }
}

/**
 * Save Camb AI diarization data to a JSON file.
 * @param streamId - Unique stream identifier
 * @param data - Diarization response data
 * @param outputDir - Directory to save the file
 * @returns Path to the saved file
 */
export async function saveCambAiDiarizationFile(streamId: string, data: any, outputDir: string): Promise<string> {
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  const filePath = path.join(outputDir, `${streamId}.json`);
  await fs.promises.writeFile(filePath, JSON.stringify(data, null, 2), "utf-8");
  logger.info(`Camb AI diarization saved to ${filePath}`);
  return filePath;
}

/**
 * Isolate foreground voice from background using Camb AI audio separation.
 * Flow: POST /audio-separation → poll → fetch foreground_audio_url → download → convert to WAV.
 * @param audioFilePath - Path to the input audio file
 * @param outputPath - Path where the isolated WAV will be saved
 * @returns Promise resolving to outputPath
 */
export async function isolateAudio(audioFilePath: string, outputPath: string): Promise<string> {
  if (!CAMB_API_KEY) {
    throw new Error("CAMB_API_KEY not set. Cannot perform Camb AI audio isolation.");
  }
  if (!fs.existsSync(audioFilePath)) {
    throw new Error(`Audio file not found: ${audioFilePath}`);
  }

  logger.info(`Starting Camb AI audio isolation for: ${audioFilePath}`);

  const outputDir = path.dirname(outputPath);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Step 1: submit
  const fileBuffer = fs.readFileSync(audioFilePath);
  const fileName = path.basename(audioFilePath);
  const formData = new FormData();
  formData.append("media_file", new Blob([fileBuffer]), fileName);

  const submitRes = await fetch(`${CAMB_API_BASE_URL}/audio-separation`, {
    method: "POST",
    headers: { "x-api-key": CAMB_API_KEY },
    body: formData,
  });
  if (!submitRes.ok) {
    throw new Error(`Camb /audio-separation failed (${submitRes.status}): ${await submitRes.text()}`);
  }
  const submitData = await submitRes.json();
  const taskId = String(submitData.task_id ?? "");
  if (!taskId) throw new Error(`Camb /audio-separation missing task_id: ${JSON.stringify(submitData)}`);
  logger.info(`Camb AI isolation submitted: taskId=${taskId}`);

  // Step 2: poll
  let runId: number | null = null;
  for (let attempt = 0; attempt < MAX_POLL_ATTEMPTS; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
    const pollRes = await fetch(`${CAMB_API_BASE_URL}/audio-separation/${taskId}`, {
      headers: { "x-api-key": CAMB_API_KEY },
    });
    if (!pollRes.ok) {
      throw new Error(`Camb isolation poll failed (${pollRes.status}): ${await pollRes.text()}`);
    }
    const pollData = await pollRes.json();
    const status = String(pollData.status ?? "").toUpperCase();
    if (status === "SUCCESS") {
      runId = pollData.run_id;
      break;
    }
    if (["ERROR", "TIMEOUT", "PAYMENT_REQUIRED"].includes(status)) {
      throw new Error(`Camb audio-separation failed: status=${status}, message=${pollData.message}`);
    }
  }
  if (runId == null) {
    throw new Error(`Camb audio-separation timed out after ${MAX_POLL_ATTEMPTS} attempts`);
  }
  logger.info(`Camb AI isolation completed: runId=${runId}`);

  // Step 3: fetch foreground URL
  const resultRes = await fetch(`${CAMB_API_BASE_URL}/audio-separation-result/${runId}`, {
    headers: { "x-api-key": CAMB_API_KEY },
  });
  if (!resultRes.ok) {
    throw new Error(`Camb /audio-separation-result failed (${resultRes.status}): ${await resultRes.text()}`);
  }
  const resultData = await resultRes.json();
  const fgUrl = resultData.foreground_audio_url;
  if (!fgUrl) throw new Error(`Camb separation-result missing foreground_audio_url: ${JSON.stringify(resultData)}`);

  // Step 4: download foreground and convert to WAV
  const dlRes = await fetch(fgUrl);
  if (!dlRes.ok) throw new Error(`Camb foreground download failed (${dlRes.status})`);

  const tempPath = `${outputPath}.tmp`;
  const arrayBuffer = await dlRes.arrayBuffer();
  fs.writeFileSync(tempPath, Buffer.from(arrayBuffer));

  await new Promise<void>((resolve, reject) => {
    const ffmpeg = spawn("ffmpeg", ["-y", "-i", tempPath, outputPath], {
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stderr = "";
    ffmpeg.stderr?.on("data", (data: Buffer) => {
      stderr += data.toString();
    });
    ffmpeg.on("close", (code) => {
      try {
        fs.unlinkSync(tempPath);
      } catch {
        /* ignore */
      }
      if (code === 0) {
        logger.info(`Camb AI isolation completed. Output saved to: ${outputPath} (converted to WAV)`);
        resolve();
      } else {
        reject(new Error(`ffmpeg failed (${code}): ${stderr.slice(-500)}`));
      }
    });
    ffmpeg.on("error", (err) => {
      try {
        fs.unlinkSync(tempPath);
      } catch {
        /* ignore */
      }
      reject(err);
    });
  });

  return outputPath;
}
