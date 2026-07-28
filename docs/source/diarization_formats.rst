Diarization Formats
===================

The ASD, Controller, and Direct clients accept a diarization file via
``--diarization-file`` and route it to a parser via
``--diarization-format``. Five formats are supported, each tied to a
specific provider or producer.

Choosing a format
-----------------

Pick the parser that matches the producer of your file. Auto-detection is
deliberately not supported: the JSON formats overlap on field names but
differ on time units and granularity, so a wrong-format pick silently
yields wrong segmentation rather than an error.

Format comparison
-----------------

.. list-table::
   :header-rows: 1
   :widths: 16 16 8 14 14 14 22

   * - Flag
     - Source
     - File
     - Granularity
     - Time fields
     - Speaker label
     - Top-level keys
   * - ``flat``
     - ASD-native flat list
     - JSON
     - segment
     - ``start_time`` / ``end_time`` (int ms)
     - integer
     - top-level list of segment dicts
   * - ``elevenlabs-scribe``
     - ElevenLabs STT (Scribe) — ``speech_to_text.convert``
     - JSON
     - word (filtered + merged into runs of same-speaker words)
     - ``start`` / ``end`` (float seconds)
     - ``"speaker_0"`` (zero-based, underscore)
     - ``text``, ``language_code``, ``words[]``
   * - ``elevenlabs-dubbing-api``
     - ElevenLabs Dubbing Transcript API — ``dubbing.transcripts.get(format_type="json")``
     - JSON
     - utterance (already pre-merged by provider)
     - ``start_s`` / ``end_s`` (float seconds, note ``_s`` suffix)
     - ``"speaker_0"`` (zero-based, underscore)
     - ``language``, ``utterances[]``
   * - ``elevenlabs-studio``
     - ElevenLabs Dubbing Studio web UI export
     - CSV
     - segment
     - ``start_time`` / ``end_time`` as ``HH:MM:SS,mmm`` strings
     - ``"Speaker 1"`` (one-based, space-separated)
     - CSV columns ``speaker``, ``start_time``, ``end_time``, ``transcription``, ``translation``
   * - ``camb``
     - Camb AI transcription (source language) or dubbing (target language) JSON
     - JSON
     - segment
     - ``start`` / ``end`` (float seconds)
     - ``"SPEAKER_0"`` (zero-based) or ``"Speaker 1"`` (one-based)
     - top-level list of segment dicts (or ``transcript`` field on a dub-result payload)

Notes on the three ElevenLabs formats
-------------------------------------

ElevenLabs publishes diarization through three distinct surfaces, each with
its own schema. They are kept as three separate parsers so callers must opt
in explicitly:

* **Scribe (STT)** is word-level with float-second timestamps and a
  ``language_code`` field. The parser filters ``type == "word"`` entries
  and merges consecutive same-speaker words into single segments.
* **Dubbing Transcript API** is utterance-level (already merged by
  ElevenLabs), uses ``start_s`` / ``end_s`` (note the ``_s`` suffix), and
  uses ``language`` (no ``_code`` suffix).
* **Studio** is a CSV export with ``HH:MM:SS,mmm`` timestamps and
  one-based ``"Speaker 1"`` labels — completely different file format and
  speaker convention.

Producers
---------

Helper scripts that generate compatible diarization files:

* ``scripts/elevenlabs/diarize.py`` → ``elevenlabs-scribe`` (source language)
* ``scripts/elevenlabs/s2s_infer.py`` (with ``--transcript-format json``) →
  ``elevenlabs-dubbing-api`` (source or target language, selected via
  ``--source-transcript-output-file`` / ``--target-transcript-output-file``)
* ``scripts/camb/diarize.py`` → ``camb`` (source language; Camb AI
  Transcription API)
* ``scripts/camb/s2s_infer.py`` (with ``--transcript-format json``) →
  ``camb`` (**target language only** — Camb AI's dubbing API does not
  expose a source-language transcript)

The Studio CSV is exported manually from the ElevenLabs Dubbing Studio
web UI. The flat JSON format has no producer script in this repository:
it directly mirrors the ASD request schema (``AudioSegmentInfo`` —
``start_time`` / ``end_time`` in integer milliseconds, integer
``speaker_id``, optional ``word`` / ``language_code``) and is intended
for hand-authored or externally generated files. Note that the ASD
client's ``--output-speaker-info`` flag writes a per-frame speaker-info
CSV (bounding boxes and speaking flags consumed by the LipSync client),
which is not a diarization input and cannot be replayed as one.

.. note::

   ASD typically expects diarization timestamps and tokens that align
   with the **source** audio. For Camb AI, that means generating
   diarization with ``scripts/camb/diarize.py`` (Transcription API), not
   ``scripts/camb/s2s_infer.py`` (Dubbing API), since the latter only
   returns target-language segments.

Default
-------

All three clients (ASD, Controller, Direct) default
``--diarization-format`` to ``elevenlabs-scribe``.
