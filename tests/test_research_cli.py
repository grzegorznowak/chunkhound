#!/usr/bin/env python3
"""
End-to-end tests for the `chunkhound research` CLI command.

These tests execute the actual CLI command without mocks to verify:
1. CLI argument parsing and validation
2. Error handling for missing database
3. Error handling for missing LLM configuration
4. Help output and argument structure
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.utils.windows_subprocess import get_safe_subprocess_env


class TestResearchCLI:
    """Test the chunkhound research CLI command end-to-end."""

    def test_research_help(self):
        """Test that research help command works."""
        result = subprocess.run(
            ["uv", "run", "chunkhound", "research", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
            env=get_safe_subprocess_env(),
        )

        assert result.returncode == 0, f"Help command failed: {result.stderr}"
        assert "research" in result.stdout.lower(), "Help should mention research"
        assert "query" in result.stdout.lower(), "Help should mention query argument"
        assert "path" in result.stdout.lower(), "Help should mention path argument"
        assert "--llm-" in result.stdout, "Help should show LLM configuration options"

    def test_research_no_database_error(self, clean_environment):
        """Test research command when database doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_dir = Path(temp_dir) / "empty_project"
            empty_dir.mkdir()

            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "chunkhound",
                    "research",
                    "How does the system work?",
                    "--config",
                    "/nonexistent/config.json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                env=get_safe_subprocess_env(),
                cwd=empty_dir,
            )

            # Should exit with error when no database exists
            assert result.returncode != 0, "Should fail when no database exists"
            # Error message may be in stdout or stderr
            error_output = result.stderr + result.stdout
            assert error_output, "Should have error message"

    def test_missing_query(self):
        """Test handling when query is missing."""
        result = subprocess.run(
            ["uv", "run", "chunkhound", "research"],
            capture_output=True,
            text=True,
            timeout=5,
            env=get_safe_subprocess_env(),
        )

        assert result.returncode != 0, "Should fail when query is missing"
        # Should show help or error about missing query
        error_output = result.stderr + result.stdout
        assert error_output, "Should have error message"

    def test_invalid_arguments(self):
        """Test handling of invalid arguments."""
        # Test invalid flag
        result = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "research",
                "test query",
                "--invalid-flag",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            env=get_safe_subprocess_env(),
        )

        assert result.returncode != 0, "Should fail with invalid argument"


class TestResearchCLIRequirements:
    """Test research command configuration requirements."""

    def test_research_requires_config(self, clean_environment):
        """Test that research fails gracefully without proper configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "test_project"
            project_dir.mkdir()

            # Create a minimal file to index
            test_file = project_dir / "test.py"
            test_file.write_text("def hello(): pass")

            # Index without embeddings (won't work for research but tests error handling)
            index_result = subprocess.run(
                ["uv", "run", "chunkhound", "index", ".", "--no-embeddings"],
                capture_output=True,
                text=True,
                timeout=10,
                env=get_safe_subprocess_env(),
                cwd=project_dir,
            )

            # Try research without LLM config
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "chunkhound",
                    "research",
                    "How does this code work?",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                env=get_safe_subprocess_env(),
                cwd=project_dir,
            )

            # Should fail due to missing configuration (config file or LLM/embedding setup)
            assert result.returncode != 0, "Should fail without proper config"
            error_output = result.stderr + result.stdout
            # Should mention either config file (.chunkhound.json), LLM, or embedding configuration needed
            assert (
                ".chunkhound.json" in error_output
                or "config" in error_output.lower()
                or "llm" in error_output.lower()
                or "embedding" in error_output.lower()
            ), f"Error should mention missing configuration, got: {error_output}"

    def test_research_with_explicit_path(self, clean_environment):
        """Test research command with explicit path argument."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "test_project"
            project_dir.mkdir()

            # Create a minimal file to index
            test_file = project_dir / "test.py"
            test_file.write_text("def hello(): pass")

            # Index without embeddings
            index_result = subprocess.run(
                ["uv", "run", "chunkhound", "index", str(project_dir), "--no-embeddings"],
                capture_output=True,
                text=True,
                timeout=10,
                env=get_safe_subprocess_env(),
            )

            # Try research with explicit path (should fail due to missing config, but validates path arg works)
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "chunkhound",
                    "research",
                    "How does this code work?",
                    str(project_dir),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                env=get_safe_subprocess_env(),
            )

            # Should fail due to missing config (proves path arg was parsed)
            assert result.returncode != 0, "Should fail without proper config"
            error_output = result.stderr + result.stdout
            # The path argument was accepted; error should be about config/llm/embedding
            assert (
                ".chunkhound.json" in error_output
                or "config" in error_output.lower()
                or "llm" in error_output.lower()
                or "embedding" in error_output.lower()
            ), f"Error should mention missing configuration, got: {error_output}"


class TestResearchDepthFlags:
    """Test research CLI depth mode flags (--shallow and --deep)."""

    def test_shallow_flag_in_help(self):
        """Test that --shallow flag appears in help."""
        result = subprocess.run(
            ["uv", "run", "chunkhound", "research", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
            env=get_safe_subprocess_env(),
        )

        assert result.returncode == 0, f"Help command failed: {result.stderr}"
        assert "--shallow" in result.stdout, "Help should show --shallow flag"
        assert (
            "quick insights" in result.stdout.lower()
            or "shallow" in result.stdout.lower()
        ), "Help should describe shallow mode"

    def test_deep_flag_in_help(self):
        """Test that --deep flag appears in help."""
        result = subprocess.run(
            ["uv", "run", "chunkhound", "research", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
            env=get_safe_subprocess_env(),
        )

        assert result.returncode == 0, f"Help command failed: {result.stderr}"
        assert "--deep" in result.stdout, "Help should show --deep flag"
        assert (
            "comprehensive" in result.stdout.lower()
            or "architectural" in result.stdout.lower()
        ), "Help should describe deep mode"

    def test_shallow_and_deep_mutually_exclusive(self):
        """Test that --shallow and --deep cannot be used together."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "research",
                "test query",
                "--shallow",
                "--deep",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            env=get_safe_subprocess_env(),
        )

        # Should fail because flags are mutually exclusive
        assert result.returncode != 0, "Should fail when both flags are used"
        error_output = result.stderr + result.stdout
        assert (
            "mutually exclusive" in error_output.lower()
            or "not allowed" in error_output.lower()
        ), f"Should indicate mutual exclusivity, got: {error_output}"

    def test_shallow_flag_accepted(self):
        """Test that --shallow flag is accepted by parser."""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_dir = Path(temp_dir) / "empty_project"
            empty_dir.mkdir()

            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "chunkhound",
                    "research",
                    "test query",
                    "--shallow",
                    "--config",
                    "/nonexistent/config.json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                env=get_safe_subprocess_env(),
                cwd=empty_dir,
            )

            # Should fail for other reasons (no database), but flag should be accepted
            # If flag wasn't accepted, we'd get "unrecognized arguments" error
            error_output = result.stderr + result.stdout
            assert (
                "unrecognized arguments: --shallow" not in error_output
            ), "Flag --shallow should be recognized"

    def test_deep_flag_accepted(self):
        """Test that --deep flag is accepted by parser."""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_dir = Path(temp_dir) / "empty_project"
            empty_dir.mkdir()

            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "chunkhound",
                    "research",
                    "test query",
                    "--deep",
                    "--config",
                    "/nonexistent/config.json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                env=get_safe_subprocess_env(),
                cwd=empty_dir,
            )

            # Should fail for other reasons (no database), but flag should be accepted
            # If flag wasn't accepted, we'd get "unrecognized arguments" error
            error_output = result.stderr + result.stdout
            assert (
                "unrecognized arguments: --deep" not in error_output
            ), "Flag --deep should be recognized"
