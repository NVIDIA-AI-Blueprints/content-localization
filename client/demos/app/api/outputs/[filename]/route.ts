/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { NextRequest } from "next/server";
import path from "path";
import { nextFileServe } from "@/app/api/utils/nextFileServe";

export async function GET(request: NextRequest, { params }: { params: Promise<{ filename: string }> }) {
  const { filename } = await params;

  const OUTPUT_DIR = process.env.OUTPUT_DIR
    ? `${process.env.OUTPUT_DIR}/outputs`
    : path.join(process.cwd(), "public", "outputs");

  const filePath = path.join(OUTPUT_DIR, path.basename(filename));

  return nextFileServe(filePath, request);
}
