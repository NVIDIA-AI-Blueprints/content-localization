/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { NextResponse } from "next/server";
import { getEnablePreprocessingFromEnv } from "../../utils/audioProcessing";

const SUPPORTED_S2S_SERVICES = ["EL_DUBBING", "CAMB_DUBBING"] as const;
type S2sService = (typeof SUPPORTED_S2S_SERVICES)[number];

type LanguageConfig = {
  supportedSourceLanguages: { code: string; label: string }[];
  supportedTargetLanguages: { code: string; label: string }[];
  defaultSourceLanguage: string;
  defaultTargetLanguage: string;
};

const isLanguageSupported = (languages: { code: string; label: string }[], language: string): boolean => {
  return languages.some((l) => l.code === language);
};

const resolveS2sService = (rawService: string | undefined): S2sService => {
  const service = rawService?.trim() || "EL_DUBBING";
  if (!SUPPORTED_S2S_SERVICES.includes(service as S2sService)) {
    throw new Error(`Unsupported S2S_SERVICE=${service}. Expected one of: ${SUPPORTED_S2S_SERVICES.join(", ")}.`);
  }
  return service as S2sService;
};

const getLanguageConfig = (s2sService: S2sService): LanguageConfig => {
  if (s2sService === "EL_DUBBING") {
    return {
      supportedSourceLanguages: [
        { code: "auto", label: "Auto detect" },
        { code: "nl", label: "Dutch (nl-NL)" },
        { code: "en", label: "English (en-US)" },
        { code: "fr", label: "French (fr-FR)" },
        { code: "de", label: "German (de-DE)" },
        { code: "es", label: "Spanish (es-ES)" },
      ],
      supportedTargetLanguages: [
        { code: "nl", label: "Dutch (nl-NL)" },
        { code: "en", label: "English (en-US)" },
        { code: "fr", label: "French (fr-FR)" },
        { code: "de", label: "German (de-DE)" },
        { code: "es", label: "Spanish (es-ES)" },
      ],
      defaultSourceLanguage: "auto",
      defaultTargetLanguage: "de",
    };
  }

  // CambAI uses numeric language IDs as strings.
  // Scoped to the same languages as ElevenLabs for now.
  const cambLanguages = [
    { code: "22", label: "Dutch (nl-NL)" },
    { code: "1", label: "English (en-US)" },
    { code: "25", label: "French (fr-FR)" },
    { code: "26", label: "German (de-DE)" },
    { code: "54", label: "Spanish (es-ES)" },
  ];

  return {
    supportedSourceLanguages: cambLanguages,
    supportedTargetLanguages: [...cambLanguages],
    defaultSourceLanguage: "1",
    defaultTargetLanguage: "26",
  };
};

export async function GET() {
  try {
    const s2sService = resolveS2sService(process.env.S2S_SERVICE);
    const { supportedSourceLanguages, supportedTargetLanguages, defaultSourceLanguage, defaultTargetLanguage } =
      getLanguageConfig(s2sService);
    const enablePreprocessing = getEnablePreprocessingFromEnv();

    const configMapping = {
      target_language: process.env.TARGET_LANGUAGE_LABEL,
      voice_name: process.env.VOICE_NAME,
      enable_preprocessing: enablePreprocessing,
      supported_source_languages: supportedSourceLanguages,
      supported_target_languages: supportedTargetLanguages,
      default_source_language: isLanguageSupported(
        supportedSourceLanguages,
        process.env.DEFAULT_SOURCE_LANGUAGE || defaultSourceLanguage,
      )
        ? process.env.DEFAULT_SOURCE_LANGUAGE
        : defaultSourceLanguage,
      default_target_language: isLanguageSupported(
        supportedTargetLanguages,
        process.env.DEFAULT_TARGET_LANGUAGE || defaultTargetLanguage,
      )
        ? process.env.DEFAULT_TARGET_LANGUAGE
        : defaultTargetLanguage,
    };

    return NextResponse.json(configMapping);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to load config";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
