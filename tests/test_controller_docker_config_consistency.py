# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for push-only controller docker/config wiring."""

import unittest
from pathlib import Path

import pytest

from controller_service.service import ControllerService

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
class TestControllerDockerConfigConsistency(unittest.TestCase):
    """Regression checks for push-only controller runtime wiring."""

    def test_service_mode_removed_from_runtime_wiring(self) -> None:
        """Compose and entrypoint shell do not reference removed service-mode."""
        compose_text = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        shell_text = (REPO_ROOT / "src/docker_entrypoints/controller/entrypoint.sh").read_text(
            encoding="utf-8"
        )
        default_env = (REPO_ROOT / "configs/elevenlabs.env").read_text(encoding="utf-8")

        self.assertNotIn("CONTROLLER_SERVICE_MODE", compose_text)
        self.assertNotIn("--service-mode", shell_text)
        self.assertNotIn("CONTROLLER_SERVICE_MODE", default_env)

        # bypass_asd is now per-request, not deployment-time
        self.assertNotIn("CONTROLLER_NO_ASD", compose_text)
        self.assertNotIn("CONTROLLER_NO_ASD", default_env)
        self.assertNotIn("--no-asd", shell_text)

    def test_argsfactory_omits_service_mode(self) -> None:
        """Controller parser omits removed service-mode argument."""
        parser = ControllerService.argsfactory()
        all_options = {
            option for action in parser._actions for option in getattr(action, "option_strings", [])
        }
        self.assertNotIn("--service-mode", all_options)
        self.assertNotIn("--no-asd", all_options)
        self.assertIn("--asd-server", all_options)

    def test_lipsync_input_queue_timeout_is_wired_to_all_configs(self) -> None:
        """LipSync input queue timeout is passed through every compose config."""
        variable = "NV_AI4M_LS_INPUT_QUEUE_TIMEOUT_S"
        compose_text = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn(f"{variable}=${{{variable}:-5}}", compose_text)

        for env_file in ("configs/elevenlabs.env", "configs/camb.env", "configs/debug.env"):
            with self.subTest(env_file=env_file):
                env_text = (REPO_ROOT / env_file).read_text(encoding="utf-8")
                self.assertIn(f"{variable}=5", env_text)

    def test_lipsync_env_variables_use_consistent_prefix(self) -> None:
        """LipSync docker env variables use LIPSYNC rather than LIP_SYNC."""
        compose_text = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        deploy_text = (REPO_ROOT / "scripts/nims/deploy_lipsync.sh").read_text(encoding="utf-8")

        self.assertNotIn("LIP_SYNC_", compose_text)
        self.assertIn("image: ${LIPSYNC_IMAGE}", compose_text)
        self.assertIn("NIM_TAGS_SELECTOR=${LIPSYNC_NIM_TAGS_SELECTOR}", compose_text)

        self.assertNotIn("LIP_SYNC_", deploy_text)
        self.assertIn("LIPSYNC_IMAGE_DEFAULT", deploy_text)
        self.assertIn("LIPSYNC_NIM_TAGS_SELECTOR", deploy_text)

        for env_file in ("configs/elevenlabs.env", "configs/camb.env", "configs/debug.env"):
            with self.subTest(env_file=env_file):
                env_text = (REPO_ROOT / env_file).read_text(encoding="utf-8")
                self.assertNotIn("LIP_SYNC_", env_text)
                self.assertIn("LIPSYNC_IMAGE=", env_text)
                self.assertIn("LIPSYNC_NIM_TAGS_SELECTOR=", env_text)
