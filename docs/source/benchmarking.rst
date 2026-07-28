.. _benchmarking:

Performance Benchmarking
========================

The performance benchmark matrix measures per-stage and end-to-end pipeline
latency for the content localization pipeline. It decomposes total pipeline
time into three independently observable components:

- **S2S time** — standalone Speech-to-Speech client latency, per asset.
- **ASD + LipSync time** — ``bypass_s2s`` mode (GPU stages only, no S2S).
- **Full e2e** — S2S + ASD + LipSync running concurrently.

Each run is an isolated one-shot controller request, matching production
usage. Comparing the bypass and full-pipeline numbers reveals how much of
end-to-end latency is attributable to S2S vs GPU processing.

----

Prerequisites
-------------

1. Repo checked out with the Python virtual environment built (``.venv/`` present).
2. ``.env`` at the repo root containing API key(s) for the S2S backend under test.
3. Benchmark video assets present and listed in ``scripts/perf/assets.manifest``.
   Edit the manifest to point at your own ``.mp4`` files; missing files are
   skipped with a warning.
4. Pipeline services running for the backend under test (see :ref:`deployment`).

The benchmark scripts activate ``.venv``, source ``.env``, and set
``PYTHONPATH`` automatically — run them from anywhere in the repo.

----

Quick Start
-----------

**Step 1 — Run the ElevenLabs backend**

Start the full stack with the ElevenLabs S2S backend:

.. code-block:: bash

   docker compose --profile controller-third-party-s2s \
       --env-file configs/elevenlabs.env --env-file .env up -d

Wait until all four services are healthy (controller on port 50056, S2S on
50050, ASD on 50055, LipSync on 50054), then run:

.. code-block:: bash

   bash scripts/perf/run_perf_matrix.sh --config el      # EL e2e + S2S-EL latency
   bash scripts/perf/run_perf_matrix.sh --config bypass  # ASD + LipSync only

**Step 2 — (Optional) Run the Camb AI backend**

The ``bypass`` config is backend-independent and only needs to run once.
To also measure Camb AI end-to-end latency:

.. code-block:: bash

   docker compose --profile controller-third-party-s2s \
       --env-file configs/camb.env --env-file .env up -d speech-to-speech

   bash scripts/perf/run_perf_matrix.sh --config camb    # Camb e2e + S2S-Camb

**Step 3 — Aggregate into the report**

.. code-block:: bash

   python scripts/perf/aggregate_perf.py --in-dir outputs/perf

This produces three output files:

- ``outputs/perf/perf_comparison.csv`` — machine-readable flat table.
- ``outputs/perf/perf_comparison.md`` — Markdown report with summary and FPS tables.
- ``outputs/perf/perf_comparison.html`` — interactive HTML dashboard (see below).

Pass ``--no-html`` to skip the HTML dashboard if not needed.

----

Output Layout
-------------

::

   outputs/perf/
     el/
       combine/<asset>/batch_processing_report.json     # EL e2e, merged-per-speaker
       combine/<asset>/fps.json                         # scraped ASD/LipSync FPS
       per_segment/<asset>/batch_processing_report.json # EL e2e, per-segment diarization
       s2s/<asset>.json                                 # standalone S2S-EL latency
     camb/   ... (same shape as el/)
     bypass/
       combine/<asset>/batch_processing_report.json     # ASD + LipSync only
       combine/<asset>/fps.json
       per_segment/<asset>/...
     perf_comparison.csv
     perf_comparison.md
     perf_comparison.html

----

Understanding the Output
------------------------

RT Factor
~~~~~~~~~

The **real-time (RT) factor** is ``pipeline_time / video_duration``. Values
below 1.0 indicate faster-than-real-time processing; lower is better.

- The **full-pipeline RT factor** reflects the S2S API round-trip latency
  more than GPU throughput, especially for short clips where the fixed API
  overhead dominates.
- The **bypass RT factor** is the GPU floor (ASD + LipSync only) and is the
  better signal for hardware scaling.

S2S Contribution
~~~~~~~~~~~~~~~~

``s2s_contribution_secs = e2e(full) - e2e(bypass)`` (same asset and
diarization mode) estimates the S2S share of end-to-end latency. Because S2S
and ASD+LipSync run concurrently, this value approaches zero for videos long
enough that ASD+LipSync dominates — the S2S overhead is fully absorbed.

e2e FPS
~~~~~~~

**e2e FPS** = ``source_frames / pipeline_wall_time``. This measures true
end-to-end throughput including gRPC transport, preprocessing, audio sync
waits, and queuing — everything observable from outside the pipeline.

NIM FPS
~~~~~~~

**ASD FPS** and **LipSync FPS** are scraped from the NIM container logs and
measure raw inference throughput inside the NIM. The ``e2e FPS / NIM FPS``
ratio reveals how much overhead the pipeline adds on top of raw inference;
values near 100% mean negligible overhead.

ASD+LipSync Time
~~~~~~~~~~~~~~~~

For the **bypass** config, the pipeline time equals the ASD+LipSync wall
time — the GPU floor achievable with no S2S step. Cross-check the bypass
``e2e_pipeline_secs`` against the full-pipeline value to isolate S2S impact.

----

HTML Dashboard
--------------

The interactive dashboard (``perf_comparison.html``) provides a visual
overview of the benchmark results:

- **Executive Summary** — computed bullet points highlighting key findings:
  average RT factors, videos where S2S overhead is fully absorbed, clips most
  impacted by S2S fixed overhead, and LipSync FPS comparison.
- **Notation & Definitions** — definitions of all terms used in the report.
- **Per-Video Metrics table** — grouped columns showing Pipeline Time,
  ASD FPS, LipSync FPS, and e2e FPS side-by-side for Bypass S2S and Full
  Pipeline.
- **Charts** — two side-by-side bar charts:

  - *e2e FPS — Bypass S2S vs Full Pipeline*: compares end-to-end throughput.
  - *Pipeline Time — S2S Wall vs ASD+LipSync (Bypass S2S)*: shows how S2S
    and GPU times compare to the video duration (real-time boundary).

To view the dashboard, open ``perf_comparison.html`` in any modern browser.
Chart rendering requires internet access to load Chart.js from
``cdn.jsdelivr.net``.

----

Reading the Numbers
-------------------

- **Long videos (≥ 30 s):** S2S overhead is typically absorbed by the
  concurrent ASD+LipSync processing. Full-pipeline and bypass RT factors
  converge. Use ``s2s_contribution_secs`` to confirm.

- **Short clips (< 20 s):** The fixed S2S API latency dominates. Bypass S2S
  is the more representative signal for GPU throughput at short durations.

- **ASD + LipSync FPS:** The ``overall_asd_lipsync_fps`` column in the CSV
  (``lipsync_frames / pipeline_wall_time``) should approach the NIM-reported
  ``lipsync_fps`` for bypass runs. The gap quantifies gRPC/streaming overhead.

- **Diarization granularity delta:** The Markdown report includes a table
  comparing merged-per-speaker vs per-segment diarization pipeline times.
  A positive delta means per-segment is slower (more gRPC round trips).
