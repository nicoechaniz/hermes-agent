"""Pin the five intent-composition rules to the daemoncraft-base profile.

These rules are validated in primitives_lab experiments 001-007 (vault
pages lessons-001-003-primitives-baseline.md and
lessons-004-007-primitives-second-round.md). Removing them from the
profile silently degrades cloud LLM intent-compose quality, so this
test exists as a regression gate.

The test reads the live profile config at ~/.hermes/profiles/. If the
profile isn't installed (e.g., on CI without the user's home dir),
the test skips rather than fails.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


PROFILE_CONFIG = Path.home() / ".hermes/profiles/daemoncraft-base/config.yaml"


@pytest.fixture
def profile_config() -> dict:
    if not PROFILE_CONFIG.exists():
        pytest.skip(f"daemoncraft-base profile not installed at {PROFILE_CONFIG}")
    return yaml.safe_load(PROFILE_CONFIG.read_text())


def test_agent_system_prompt_is_present(profile_config):
    sp = profile_config.get("agent", {}).get("system_prompt", "")
    assert sp, "agent.system_prompt is missing or empty"


def test_all_five_intent_rules_present(profile_config):
    sp = profile_config["agent"]["system_prompt"]
    for n in (1, 2, 3, 4, 5):
        assert f"RULE {n}" in sp, f"RULE {n} marker missing from system_prompt"


def test_rule_text_carries_concrete_examples(profile_config):
    """If an engineer trims the rules to bullets, the model loses the
    examples that make the rules actionable. Pin the load-bearing strings."""
    sp = profile_config["agent"]["system_prompt"]
    expected_phrases = [
        "English imperative",          # rule 1 keyword
        "explicit non-bot coordinates", # rule 2 keyword
        "numbered stages",              # rule 3 keyword
        "delegate conditionals",        # rule 4 keyword
        "explicit username",            # rule 5 keyword
        "(5, 65, 37)",                  # concrete-coords example in rule 2
        "Step 1:",                      # numbered-stages example in rule 3
    ]
    for phrase in expected_phrases:
        assert phrase in sp, f"missing key phrase: {phrase!r}"


def test_recovery_section_documents_auto_retry(profile_config):
    """Pipeline 2 changes the tool to auto-retry on failure. The system
    prompt must tell the cloud LLM not to micro-manage retries (or it
    will create double-recovery loops). Pin that note."""
    sp = profile_config["agent"]["system_prompt"]
    assert "synchronous retry" in sp or "happens automatically" in sp, \
        "system_prompt must inform cloud LLM that recovery is automatic"
