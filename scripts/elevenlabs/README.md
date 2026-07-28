# scripts/elevenlabs

Standalone ElevenLabs scripts. Authenticate via the `ELEVENLABS_API_KEY`
environment variable (load `.env` before running). Each script is self-contained
and is invoked directly with `python scripts/elevenlabs/<name>.py`.

| Script | Purpose |
| --- | --- |
| `s2s_infer.py` | End-to-end dubbing via the ElevenLabs Dubbing API. Output: dubbed audio + transcript. |
| `diarize.py` | Speaker diarization via the ElevenLabs Scribe STT API. |
| `audio_isolation.py` | Voice isolation (foreground stem). |
| `stem_separation.py` | Multi-stem audio separation. |
| `invoke_e2e.sh` | Convenience wrapper around `s2s_infer.py`. |
