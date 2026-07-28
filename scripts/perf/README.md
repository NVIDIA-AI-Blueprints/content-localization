# Performance benchmark matrix

Measures end-to-end and **per-stage** latency of the content-localization
pipeline, decomposed into:

- **S2S time** — standalone Speech-to-Speech client (ElevenLabs / Camb), per asset.
- **ASD + LipSync time** — `bypass_s2s` mode (skip S2S, feed pre-translated audio
  straight to ASD → LipSync), per asset. This isolates the GPU stages.
- **Full e2e** — EL and Camb (S2S + ASD + LipSync), to verify the parts ≈ the whole.

Each cell runs as its **own one-shot controller request**, because the controller
serves a **single request at a time** (`CONTROLLER_MAX_CONCURRENCY=1` by design).

## Files

| File | Purpose |
|------|---------|
| `prepare_perf_assets.py` | Break one asset video into duration-labeled benchmark clips and write a manifest. |
| `run_perf_matrix.sh` | Run the matrix for ONE S2S backend (`--config el\|camb\|bypass`). One video per process, controller idle between requests. |
| `run_perf_analysis.sh` | Run repeated merged-per-speaker benchmark jobs on one machine, writing to a directory named after that machine. |
| `aggregate_perf.py` | Merge all `perf/*` runs into a CSV + Markdown + HTML comparison with derived per-stage columns. |
| `aggregate_repeated_perf.py` | Combine repeated-run outputs from one or more machines (`--dirs`, machine label = directory basename), optionally add a baseline side-by-side CSV as run 0, and emit averaged CSV/Markdown records. |
| `build_combined_perf_report.py` | Build the self-contained CLBP Perf analysis HTML from the side-by-side CSV; machine columns are discovered from the CSV header. |
| `assets.manifest` | List of benchmark videos (length-tagged). Edit to taste; missing files are skipped. |

## Prerequisites (on the target machine)

1. Repo checked out with the Python venv built: `.venv/` present.
2. `.env` at the repo root with `ELEVENLABS_API_KEY` and `CAMB_API_KEY`.
3. Benchmark assets present (see `assets.manifest`). Add your own video paths to
   the manifest; missing files are skipped with a warning.
4. The pipeline services up for the backend under test (see below).

> The scripts `cd` to the repo root, activate `.venv`, source `.env`, and set
> `PYTHONPATH` themselves — run them from anywhere in the repo.

## Prepare benchmark clips from any asset video

Use `prepare_perf_assets.py` to make duration-labeled clips and a matching
manifest from a single source video:

```bash
source .venv/bin/activate && \
python scripts/perf/prepare_perf_assets.py \
  --input-video assets/<video>.mp4 \
  --out-dir outputs/perf_segments/<run-name> \
  --manifest outputs/perf_segments/<run-name>/assets.manifest
```

Defaults create `10s`, `20s`, `30s`, `1min`, `2min`, `5min`, `10min`, and full
clips, skipping clips longer than the source video. Use `--audio-codec aac` when
the target comparison machine needs AAC-normalized MP4 inputs. The manifest tags
are duration-only labels so reports can display clip length without source-title
metadata.

## ⚠️ Operating rules (the controller is single-worker)

- **One request at a time.** Never run two pipeline invocations concurrently.
- **Never health-probe the controller while it is serving** — it shares the one
  worker and will `DEADLINE_EXCEEDED`.
- **Never `kill -9` a run mid-request** — it leaves the request stuck on the
  single worker and wedges the controller (recovery = restart the `controller`
  container). Let runs finish, or use the built-in timeouts.
- A controller-logged `asd:50055 health check failed: DEADLINE_EXCEEDED` almost
  always means the *controller* is busy/wedged, **not** that ASD is down — the
  NIMs answer gRPC health in ~4 ms; verify directly before blaming a NIM.

## How to run the full matrix

### 1. ElevenLabs backend (covers EL e2e, S2S-EL, and backend-independent bypass)

```bash
# bring up the third-party-S2S stack with the ElevenLabs backend
docker compose --profile controller-third-party-s2s \
  --env-file configs/elevenlabs.env --env-file .env up -d

# wait until all four are healthy: controller(50056) s2s(50050) asd(50055) lipsync(50054)
# then run (each is internally one-video-per-process):
bash scripts/perf/run_perf_matrix.sh --config el      # EL e2e + standalone S2S-EL
bash scripts/perf/run_perf_matrix.sh --config bypass  # ASD + LipSync only
```

### 2. Camb backend (Camb e2e + S2S-Camb)

```bash
# switch the S2S service to Camb, then re-run only the S2S-dependent config
docker compose --profile controller-third-party-s2s \
  --env-file configs/camb.env --env-file .env up -d s2s
# (or restart the whole stack with configs/camb.env)

bash scripts/perf/run_perf_matrix.sh --config camb    # Camb e2e + standalone S2S-Camb
```

`bypass` is backend-independent (S2S is skipped), so it only needs to run once.

### 3. Aggregate into the report

```bash
python scripts/perf/aggregate_perf.py --in-dir outputs/perf
# writes outputs/perf/perf_comparison.csv and outputs/perf/perf_comparison.md
# also writes outputs/perf/perf_comparison.html (interactive dashboard)
```

For the CLBP Perf analysis HTML, build from the consolidated side-by-side CSV:

```bash
source .venv/bin/activate && \
python scripts/perf/build_combined_perf_report.py \
  --side-by-side-csv outputs/combined-perf-report/side_by_side_avg3.csv \
  --output-html outputs/combined-perf-report/combined_perf_report.html
```

For repeated runs, run one machine-local benchmark job per machine, writing each
machine's output to a directory named after that machine (the directory basename
becomes the machine label in reports), then aggregate the machine outputs. One
`--dirs` entry produces a single-machine report; two or more entries produce a
side-by-side comparison:

```bash
source .venv/bin/activate && \
bash scripts/perf/run_perf_analysis.sh \
  --manifest outputs/perf_segments/<run-name>/assets.manifest \
  --out-dir outputs/perf_<run-name>/<machine-name> \
  --runs 3 \
  --configs bypass,el

# repeat on each additional machine, using its own <machine-name> directory

source .venv/bin/activate && \
python scripts/perf/aggregate_repeated_perf.py \
  --dirs outputs/perf_<run-name>/<machine-name> outputs/perf_<run-name>/<other-machine> \
  --baseline-csv outputs/combined-perf-report/side_by_side_avg3.csv \
  --drop-outliers \
  --report-suffix 4run \
  --out-dir outputs/combined-perf-report-4run

source .venv/bin/activate && \
python scripts/perf/build_combined_perf_report.py \
  --side-by-side-csv outputs/combined-perf-report-4run/side_by_side_4run.csv \
  --output-html outputs/combined-perf-report-4run/combined_perf_report_4run.html \
  --subtitle "Four-run average FPS comparison, outliers excluded"
```

The combined report HTML is self-contained and embeds inline SVG bar charts. It
uses Bypass S2S and E2E / ElevenLabs column groups and calls `calc_fps` Pipeline
FPS in visible text. When repeated-run standard deviations are available, tables
show `mean +/- stddev` and FPS charts include error bars.

## Options

`run_perf_matrix.sh` flags (all optional except `--config`):

```
--config {el|camb|bypass}   required
--manifest PATH             default scripts/perf/assets.manifest
--out-dir PATH              default outputs/perf
--target-language L         default de
--source-language L         default en
--controller-server H:P     default localhost:50056
--s2s-server H:P            default localhost:50050
```

`run_perf_analysis.sh` adds:

```
--machine-label LABEL       optional machine label recorded in machine_info.json
--runs N                    default 3
--run-start N               default 1
--configs LIST              default el,bypass
--canonical-diarization-dir PATH
```

Per-request ASD/LipSync FPS are scraped from the NIM container logs after each
run. Override the container names with `ASD_CONTAINER` / `LIPSYNC_CONTAINER` env
vars (default `asd` / `lipsync`); FPS capture is skipped gracefully if `docker`
or the containers aren't available.

## Output layout

```
outputs/perf/
  el/
    combine/<asset>/batch_processing_report.json     # EL e2e, merged-per-speaker diarization
    combine/<asset>/fps.json                         # scraped ASD/LipSync frames + FPS
    per_segment/<asset>/batch_processing_report.json # EL e2e, one-chunk-per-segment diarization
    s2s/<asset>.json                                 # standalone S2S-EL latency
  camb/   ... (same shape)
  bypass/
    combine/<asset>/batch_processing_report.json     # ASD + LipSync only
    combine/<asset>/fps.json
    per_segment/<asset>/...
  perf_comparison.csv / perf_comparison.md / perf_comparison.html  # from aggregate_perf.py
```

Each `batch_processing_report.json` carries `video_duration_secs`,
`pipeline_time_secs`, and `realtime_factor` (= pipeline_time / duration). Each
S2S `<asset>.json` carries per-chunk / output-stream latency, `is_realtime`, and
wall time. Each `fps.json` carries `asd_frames/asd_fps` and
`lipsync_frames/lipsync_fps`.

`aggregate_perf.py` writes the comparison + a **diarization-granularity delta**
table + an **ASD + LipSync FPS** table (also printed to the console), with:
`asd_fps`, `lipsync_fps`, `overall_asd_lipsync_fps` (= lipsync_frames /
pipeline_time), and `overall/lipsync` %.

## Reading the numbers

- **EL/Camb e2e RT factor** is dominated by the dubbing **API round-trip** (the
  backend dubs the whole clip in one call), so it reflects API latency more than
  GPU work and won't scale linearly with clip length.
- **bypass RT factor** is the GPU floor (ASD + LipSync) and is the better
  hardware-scaling signal.
- **S2S `<asset>.json`** is the right "is the S2S stage real-time" signal
  (per-chunk streaming latency vs chunk size).
- Derived `s2s_contribution_secs = e2e − bypass` (same asset/diarization mode)
  estimates the S2S share of end-to-end latency; cross-check against the
  standalone S2S wall time.
- **ASD + LipSync FPS**: `asd_fps`/`lipsync_fps` are each stage's own throughput;
  `overall_asd_lipsync_fps` is frames ÷ pipeline wall time. For a streaming
  pipeline the **overall should approach the LipSync FPS** (the slower stage) —
  the `overall/lipsync` ratio quantifies the overlap (use the **bypass** rows;
  for el/camb the denominator also includes S2S, so the ratio is lower).
