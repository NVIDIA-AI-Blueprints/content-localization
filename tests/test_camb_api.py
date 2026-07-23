# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CambAI REST API helpers (s2s_service.camb_utils.api)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import requests

from common.audio_utils import audio_mime_type
from s2s_service.camb_utils.api import CambAltFormatPolling
from s2s_service.camb_utils.api import _confirm_upload
from s2s_service.camb_utils.api import _request_upload_url
from s2s_service.camb_utils.api import _upload_file_to_presigned_url
from s2s_service.camb_utils.api import download_output_audio_to_file
from s2s_service.camb_utils.api import find_audio_url
from s2s_service.camb_utils.api import get_alt_format_output_audio_url
from s2s_service.camb_utils.api import request_dub_alt_format
from s2s_service.camb_utils.api import submit_dub_task
from s2s_service.camb_utils.api import upload_local_file
from s2s_service.camb_utils.api import wait_for_alt_format_completion
from s2s_service.camb_utils.api import wait_for_completion

HEADERS = {"x-api-key": "test-key"}


@pytest.mark.unit
class TestAudioMimeType(unittest.TestCase):
    """Tests for common.audio_utils.audio_mime_type."""

    def test_wav_file(self) -> None:
        result = audio_mime_type(Path("audio.wav"))
        self.assertEqual(result, "audio/wav")

    def test_mp3_file(self) -> None:
        result = audio_mime_type(Path("audio.mp3"))
        self.assertEqual(result, "audio/mpeg")

    def test_unknown_extension_falls_back_to_wav(self) -> None:
        result = audio_mime_type(Path("file.xyz123"))
        self.assertEqual(result, "audio/wav")


@pytest.mark.unit
class TestRequestUploadUrl(unittest.TestCase):
    """Tests for _request_upload_url."""

    @patch("s2s_service.camb_utils.api.requests.post")
    def test_success(self, mock_post: MagicMock) -> None:
        """Successful upload URL request returns file_id, url, headers."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "file": {"file_id": "f-123"},
            "upload": {"url": "https://storage/upload", "headers": {"x-amz": "val"}},
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        file_id, url, hdrs = _request_upload_url(
            filename="clip.wav",
            content_type="audio/x-wav",
            headers=HEADERS,
        )
        self.assertEqual(file_id, "f-123")
        self.assertEqual(url, "https://storage/upload")
        self.assertEqual(hdrs, {"x-amz": "val"})

    @patch("s2s_service.camb_utils.api.requests.post")
    def test_http_error(self, mock_post: MagicMock) -> None:
        """HTTP errors should propagate."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("401")
        mock_post.return_value = mock_response

        with self.assertRaises(requests.HTTPError):
            _request_upload_url(
                filename="clip.wav",
                content_type="audio/x-wav",
                headers=HEADERS,
            )


@pytest.mark.unit
class TestUploadFileToPresignedUrl(unittest.TestCase):
    """Tests for _upload_file_to_presigned_url."""

    @patch("s2s_service.camb_utils.api.requests.put")
    def test_success(self, mock_put: MagicMock) -> None:
        """Successful upload (status 200) should not raise."""
        mock_put.return_value = MagicMock(status_code=200)

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(b"fake audio")
            tmp.flush()
            _upload_file_to_presigned_url(
                file_path=Path(tmp.name),
                upload_url="https://storage/upload",
                upload_headers={},
            )

    @patch("s2s_service.camb_utils.api.requests.put")
    def test_failure_status(self, mock_put: MagicMock) -> None:
        """Non-success status codes should raise RuntimeError."""
        mock_put.return_value = MagicMock(status_code=403, text="Forbidden")

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(b"fake audio")
            tmp.flush()
            with self.assertRaises(RuntimeError) as ctx:
                _upload_file_to_presigned_url(
                    file_path=Path(tmp.name),
                    upload_url="https://storage/upload",
                    upload_headers={},
                )
            self.assertIn("403", str(ctx.exception))


@pytest.mark.unit
class TestConfirmUpload(unittest.TestCase):
    """Tests for _confirm_upload."""

    @patch("s2s_service.camb_utils.api.requests.post")
    def test_success(self, mock_post: MagicMock) -> None:
        """Successful confirmation should not raise."""
        mock_post.return_value = MagicMock()
        mock_post.return_value.raise_for_status = MagicMock()
        _confirm_upload(file_id="f-123", headers=HEADERS)
        mock_post.assert_called_once()


@pytest.mark.unit
class TestUploadLocalFile(unittest.TestCase):
    """Tests for upload_local_file (three-step flow)."""

    @patch("s2s_service.camb_utils.api._confirm_upload")
    @patch("s2s_service.camb_utils.api._upload_file_to_presigned_url")
    @patch("s2s_service.camb_utils.api._request_upload_url")
    def test_full_flow(
        self,
        mock_request_url: MagicMock,
        mock_upload: MagicMock,
        mock_confirm: MagicMock,
    ) -> None:
        """Full upload flow returns file_id."""
        mock_request_url.return_value = ("f-456", "https://s3/upload", {})

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"fake audio data")
            tmp.flush()
            file_id = upload_local_file(
                file_path=Path(tmp.name),
                headers=HEADERS,
            )

        self.assertEqual(file_id, "f-456")
        mock_request_url.assert_called_once()
        mock_upload.assert_called_once()
        mock_confirm.assert_called_once_with(file_id="f-456", headers=HEADERS)

    def test_file_not_found(self) -> None:
        """Missing file should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            upload_local_file(
                file_path=Path("/nonexistent/file.wav"),
                headers=HEADERS,
            )


@pytest.mark.unit
class TestSubmitDubTask(unittest.TestCase):
    """Tests for submit_dub_task."""

    @patch("s2s_service.camb_utils.api.requests.post")
    def test_success(self, mock_post: MagicMock) -> None:
        """Valid response returns task_id."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"task_id": "task-abc"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        task_id = submit_dub_task(
            source_language_id=1,
            target_language_id=54,
            headers=HEADERS,
            file_id="f-123",
        )
        self.assertEqual(task_id, "task-abc")

    @patch("s2s_service.camb_utils.api.requests.post")
    def test_missing_task_id(self, mock_post: MagicMock) -> None:
        """Missing task_id should raise RuntimeError."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with self.assertRaises(RuntimeError) as ctx:
            submit_dub_task(
                source_language_id=1,
                target_language_id=54,
                headers=HEADERS,
                file_id="f-123",
            )
        self.assertIn("task_id", str(ctx.exception))


@pytest.mark.unit
class TestWaitForCompletion(unittest.TestCase):
    """Tests for wait_for_completion."""

    @patch("s2s_service.camb_utils.api.time.sleep")
    @patch("s2s_service.camb_utils.api.requests.get")
    def test_immediate_success(self, mock_get: MagicMock, mock_sleep: MagicMock) -> None:
        """Immediate SUCCESS returns run_id without sleeping."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "SUCCESS", "run_id": 42}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        run_id = wait_for_completion(
            task_id="task-abc",
            headers=HEADERS,
            max_attempts=5,
            poll_interval_seconds=1,
        )
        self.assertEqual(run_id, 42)
        mock_sleep.assert_not_called()

    @patch("s2s_service.camb_utils.api.time.sleep")
    @patch("s2s_service.camb_utils.api.requests.get")
    def test_poll_then_success(self, mock_get: MagicMock, mock_sleep: MagicMock) -> None:
        """Polling through PROCESSING then SUCCESS."""
        processing = MagicMock()
        processing.json.return_value = {"status": "PROCESSING"}
        processing.raise_for_status = MagicMock()

        success = MagicMock()
        success.json.return_value = {"status": "SUCCESS", "run_id": 99}
        success.raise_for_status = MagicMock()

        mock_get.side_effect = [processing, processing, success]

        run_id = wait_for_completion(
            task_id="task-abc",
            headers=HEADERS,
            max_attempts=10,
            poll_interval_seconds=1,
        )
        self.assertEqual(run_id, 99)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("s2s_service.camb_utils.api.time.sleep")
    @patch("s2s_service.camb_utils.api.requests.get")
    def test_error_status(self, mock_get: MagicMock, _mock_sleep: MagicMock) -> None:
        """ERROR status should raise RuntimeError."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ERROR",
            "message": "bad input",
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with self.assertRaises(RuntimeError) as ctx:
            wait_for_completion(
                task_id="task-abc",
                headers=HEADERS,
                max_attempts=5,
                poll_interval_seconds=1,
            )
        self.assertIn("ERROR", str(ctx.exception))

    @patch("s2s_service.camb_utils.api.time.sleep")
    @patch("s2s_service.camb_utils.api.requests.get")
    def test_timeout(self, mock_get: MagicMock, _mock_sleep: MagicMock) -> None:
        """Exceeding max_attempts should raise TimeoutError."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "PROCESSING"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with self.assertRaises(TimeoutError):
            wait_for_completion(
                task_id="task-abc",
                headers=HEADERS,
                max_attempts=3,
                poll_interval_seconds=1,
            )
        self.assertEqual(mock_get.call_count, 3)


@pytest.mark.unit
class TestFindAudioUrl(unittest.TestCase):
    """Tests for find_audio_url."""

    def test_nested_url(self) -> None:
        """Nested URL fields should be discovered."""
        payload = {"data": [{"result": {"download_url": "https://cdn/out.mp3"}}]}

        self.assertEqual(find_audio_url(payload), "https://cdn/out.mp3")

    def test_missing_url(self) -> None:
        """Payloads without URL fields return None."""
        self.assertIsNone(find_audio_url({"data": [{"status": "SUCCESS"}]}))


@pytest.mark.unit
class TestRequestDubAltFormat(unittest.TestCase):
    """Tests for request_dub_alt_format."""

    @patch("s2s_service.camb_utils.api.requests.post")
    def test_defaults_to_mp3(self, mock_post: MagicMock) -> None:
        """Alt-format requests should default to MP3."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"task_id": "task-alt"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        payload = request_dub_alt_format(run_id=42, language="54", headers=HEADERS)

        self.assertEqual(payload, {"task_id": "task-alt"})
        args, kwargs = mock_post.call_args
        self.assertIn("/dub-alt-format/42/54", args[0])
        self.assertEqual(kwargs["json"], {"output_format": "mp3"})

    @patch("s2s_service.camb_utils.api.requests.post")
    def test_non_object_response_raises(self, mock_post: MagicMock) -> None:
        """Non-object JSON should raise RuntimeError."""
        mock_response = MagicMock()
        mock_response.json.return_value = ["unexpected"]
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with self.assertRaises(RuntimeError):
            request_dub_alt_format(run_id=42, language="54", headers=HEADERS)


@pytest.mark.unit
class TestWaitForAltFormatCompletion(unittest.TestCase):
    """Tests for wait_for_alt_format_completion."""

    @patch("s2s_service.camb_utils.api.time.sleep")
    @patch("s2s_service.camb_utils.api.requests.get")
    def test_poll_then_success(self, mock_get: MagicMock, mock_sleep: MagicMock) -> None:
        """Polling through PENDING then SUCCESS returns terminal payload."""
        pending = MagicMock()
        pending.json.return_value = {"status": "PENDING"}
        pending.raise_for_status = MagicMock()

        success = MagicMock()
        success.json.return_value = {"status": "SUCCESS", "output_url": "https://cdn/out.mp3"}
        success.raise_for_status = MagicMock()

        mock_get.side_effect = [pending, success]

        payload = wait_for_alt_format_completion(
            task_id="task-alt",
            headers=HEADERS,
            max_attempts=5,
            poll_interval_seconds=1,
        )
        self.assertEqual(payload["output_url"], "https://cdn/out.mp3")
        mock_sleep.assert_called_once_with(1)

    @patch("s2s_service.camb_utils.api.requests.get")
    def test_error_status_raises(self, mock_get: MagicMock) -> None:
        """Terminal error statuses should raise RuntimeError."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ERROR", "message": "bad format"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with self.assertRaises(RuntimeError) as ctx:
            wait_for_alt_format_completion(
                task_id="task-alt",
                headers=HEADERS,
                max_attempts=1,
                poll_interval_seconds=1,
            )
        self.assertIn("ERROR", str(ctx.exception))


@pytest.mark.unit
class TestGetAltFormatOutputAudioUrl(unittest.TestCase):
    """Tests for get_alt_format_output_audio_url."""

    @patch("s2s_service.camb_utils.api.requests.post")
    def test_immediate_output_url(self, mock_post: MagicMock) -> None:
        """Immediate output_url should be returned."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"output_url": "https://cdn/out.mp3"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        url = get_alt_format_output_audio_url(run_id=42, language="54", headers=HEADERS)
        self.assertEqual(url, "https://cdn/out.mp3")

    @patch("s2s_service.camb_utils.api.requests.get")
    @patch("s2s_service.camb_utils.api.requests.post")
    def test_async_success_refreshes_for_output_url(
        self,
        mock_post: MagicMock,
        mock_get: MagicMock,
    ) -> None:
        """A successful async task without URL should refresh the alt-format request."""
        create_task = MagicMock()
        create_task.json.return_value = {"task_id": "task-alt"}
        create_task.raise_for_status = MagicMock()

        refreshed = MagicMock()
        refreshed.json.return_value = {"output_url": "https://cdn/out.mp3"}
        refreshed.raise_for_status = MagicMock()
        mock_post.side_effect = [create_task, refreshed]

        success = MagicMock()
        success.json.return_value = {"status": "SUCCESS"}
        success.raise_for_status = MagicMock()
        mock_get.return_value = success

        url = get_alt_format_output_audio_url(
            run_id=42,
            language="54",
            headers=HEADERS,
        )
        self.assertEqual(url, "https://cdn/out.mp3")
        self.assertEqual(mock_post.call_count, 2)

    @patch("s2s_service.camb_utils.api.wait_for_alt_format_completion")
    @patch("s2s_service.camb_utils.api.request_dub_alt_format")
    def test_custom_polling_options(
        self,
        mock_request_alt: MagicMock,
        mock_wait: MagicMock,
    ) -> None:
        """Custom polling options should be forwarded to status polling."""
        mock_request_alt.side_effect = [
            {"task_id": "task-alt"},
            {"output_url": "https://cdn/out.mp3"},
        ]
        mock_wait.return_value = {"status": "SUCCESS"}

        url = get_alt_format_output_audio_url(
            run_id=42,
            language="54",
            headers=HEADERS,
            polling=CambAltFormatPolling(
                max_attempts=3,
                poll_interval_seconds=2,
            ),
        )

        self.assertEqual(url, "https://cdn/out.mp3")
        mock_wait.assert_called_once_with(
            task_id="task-alt",
            headers=HEADERS,
            max_attempts=3,
            poll_interval_seconds=2,
        )


@pytest.mark.unit
class TestDownloadOutputAudioToFile(unittest.TestCase):
    """Tests for download_output_audio_to_file."""

    @patch("s2s_service.camb_utils.api.requests.get")
    def test_download_success(self, mock_get: MagicMock) -> None:
        """Successful download should write file content."""
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "out.mp3"
            result = download_output_audio_to_file(
                audio_url="https://cdn/dubbed.mp3",
                output_file=output,
            )
            self.assertEqual(result, output)
            self.assertTrue(output.exists())
            self.assertEqual(output.read_bytes(), b"chunk1chunk2")


if __name__ == "__main__":
    unittest.main()
