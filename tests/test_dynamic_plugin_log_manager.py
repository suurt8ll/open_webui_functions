"""Tests for plugins.pipes.dynamic_plugin_log_manager."""

import pytest

from plugins.pipes.dynamic_plugin_log_manager import _module_matches


@pytest.mark.parametrize(
    ("record_name", "target", "expected"),
    [
        # Plain module name (OWUI <= 0.11): function_<id>
        (
            "function_gemini_manifold_google_genai",
            "function_gemini_manifold_google_genai",
            True,
        ),
        # Hash-suffixed module name (OWUI >= 0.12): function_<id>_<uuid4.hex>
        (
            "function_gemini_manifold_google_genai_7451c67be6f34c9f99455f5ad593691e",
            "function_gemini_manifold_google_genai",
            True,
        ),
        # Different function ID must never match
        (
            "function_venice_image_generation",
            "function_gemini_manifold_google_genai",
            False,
        ),
        # Hash-suffixed different function ID must never match
        (
            "function_venice_image_generation_4208a8ea4dc443d799e62d5ce03b096a",
            "function_gemini_manifold_google_genai",
            False,
        ),
        # A longer unrelated ID that starts with the target as a substring
        (
            "function_gemini_manifold_google_genai_plus",
            "function_gemini_manifold_google_genai",
            False,
        ),
        # Empty / None record names
        ("", "function_gemini_manifold_google_genai", False),
        (None, "function_gemini_manifold_google_genai", False),
    ],
)
def test_module_matches(record_name, target, expected):
    assert _module_matches(record_name, target) is expected
