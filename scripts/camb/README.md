# scripts/camb

Standalone Camb AI scripts. Authenticate via the `CAMB_API_KEY` environment variable
(load `.env` before running). Each script is self-contained — argparse, HTTP calls
via `requests`, file I/O — and is invoked directly with `python scripts/camb/<name>.py`.

| Script | Purpose |
| --- | --- |
| `s2s_infer.py` | End-to-end dubbing via Camb's file-based `/apis/dub`. Output: dubbed MP3 + optional transcript (json/txt/srt/vtt). |
| `diarize.py` | Source-language diarization via `/apis/transcribe`. Output: word-timestamped JSON. |
| `audio_isolation.py` | Vocal vs background separation via `/apis/audio-separation`. Output: foreground WAV (+ optional background WAV). |
| `invoke_e2e.sh` | Convenience wrapper around `s2s_infer.py` for the controller pipeline. |
| `invoke_diarize_asd.sh` | Convenience wrapper running `diarize.py` then feeding ASD. |

See the top-level README for end-to-end usage examples.
