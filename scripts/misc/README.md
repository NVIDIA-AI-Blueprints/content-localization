# scripts/misc

Cross-cutting development utilities: environment bootstrap, log copy,
media preprocessing, and evaluation drivers. Most scripts are provider-agnostic;
`run_all_clients.sh` also invokes ElevenLabs diarization and therefore requires
ElevenLabs configuration.

| Script | Purpose |
| --- | --- |
| `setup_env.sh` | Bootstrap dev environment from a fresh Ubuntu install. |
| `copy_docker_logs.sh` | Copy Docker container logs to local files. |
| `convert_to_streamable_mp4.sh` | Re-encode video with `faststart` for streamable MP4. |
| `extract_audio_from_videos.sh` | Batch extract audio from a directory of videos. |
| `trim_audio_to_video_length.py` | Trim an audio file to match a video's duration. |
| `run_all_clients.sh` | Run all clients against a sample video; uses ElevenLabs diarization. |
| `run_evaluation.sh` | Evaluation driver across a directory of inputs. |
| `generate_dev_certs.sh` | Dev-only CA + per-service TLS certs and mTLS client cert for testing the TLS surface. |
