"""Tests for the ship-review graph creation helper and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_review as kr


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def fake_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    return str(repo)


# ---------------------------------------------------------------------------
# ReviewGraphSpec
# ---------------------------------------------------------------------------

def test_review_graph_spec_fields():
    spec = kr.ReviewGraphSpec(
        repo_path="/tmp/repo",
        base="nousmain",
        head="feat/auth",
        title="Review PR #42",
        assignee="miki",
        ready=True,
        idempotency_prefix="prefix",
        skills=["github-code-review"],
        body="Extra context",
    )
    assert spec.repo_path == "/tmp/repo"
    assert spec.base == "nousmain"
    assert spec.head == "feat/auth"
    assert spec.title == "Review PR #42"
    assert spec.assignee == "miki"
    assert spec.ready is True
    assert spec.idempotency_prefix == "prefix"
    assert spec.skills == ["github-code-review"]
    assert spec.body == "Extra context"


def test_review_graph_spec_defaults():
    spec = kr.ReviewGraphSpec(repo_path="/tmp/repo", base="main", head="feat/x", title="T")
    assert spec.assignee is None
    assert spec.ready is False
    assert spec.idempotency_prefix is None
    assert spec.skills == []
    assert spec.body is None


# ---------------------------------------------------------------------------
# Core helper tests
# ---------------------------------------------------------------------------

def test_create_review_graph_smoke(kanban_home, fake_repo):
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    assert result["created"] is True
    assert result["parent_id"].startswith("t_")
    assert len(result["reviewer_ids"]) == 3
    assert result["synthesis_id"].startswith("t_")

    # Verify synthesis is gated on reviewers
    with kb.connect() as conn:
        for rid in result["reviewer_ids"]:
            # Reviewers are parallel — they have no parents
            assert kb.parent_ids(conn, rid) == []
        synth_parents = kb.parent_ids(conn, result["synthesis_id"])
        assert set(synth_parents) == set(result["reviewer_ids"])


def test_create_review_graph_idempotent(kanban_home, fake_repo):
    r1 = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    r2 = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    assert r1["parent_id"] == r2["parent_id"]
    assert r1["reviewer_ids"] == r2["reviewer_ids"]
    assert r1["synthesis_id"] == r2["synthesis_id"]
    assert r2["created"] is False


def test_create_review_graph_idempotent_different_base_or_head(kanban_home, fake_repo):
    """Changing base or head creates a new graph."""
    r1 = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    r2 = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/other",
        repo_path=fake_repo,
    )
    r3 = kr.create_review_graph(
        title="Review PR #42",
        base="main",
        head="feat/auth",
        repo_path=fake_repo,
    )
    assert r1["parent_id"] != r2["parent_id"]
    assert r1["parent_id"] != r3["parent_id"]
    assert r2["parent_id"] != r3["parent_id"]


def test_create_review_graph_triage_by_default(kanban_home, fake_repo):
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    with kb.connect() as conn:
        for tid in [result["parent_id"], *result["reviewer_ids"], result["synthesis_id"]]:
            task = kb.get_task(conn, tid)
            assert task.status == "triage"


def test_create_review_graph_ready_mode(kanban_home, fake_repo):
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
        ready=True,
    )
    with kb.connect() as conn:
        parent = kb.get_task(conn, result["parent_id"])
        assert parent.status == "ready"
        for rid in result["reviewer_ids"]:
            task = kb.get_task(conn, rid)
            assert task.status == "ready"
        # Synthesis starts as todo because its parents (reviewers) are not done.
        synth = kb.get_task(conn, result["synthesis_id"])
        assert synth.status == "todo"


def test_create_review_graph_assignee_and_skills(kanban_home, fake_repo):
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
        assignee="miki",
        skills=["github-code-review"],
    )
    with kb.connect() as conn:
        for tid in [result["parent_id"], *result["reviewer_ids"], result["synthesis_id"]]:
            task = kb.get_task(conn, tid)
            assert task.assignee == "miki"
            assert task.skills == ["github-code-review"]


def test_create_review_graph_workspace_is_dir(kanban_home, fake_repo):
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    with kb.connect() as conn:
        for tid in [result["parent_id"], *result["reviewer_ids"], result["synthesis_id"]]:
            task = kb.get_task(conn, tid)
            assert task.workspace_kind == "dir"
            assert task.workspace_path == str(Path(fake_repo).resolve())


def test_create_review_graph_body_appended(kanban_home, fake_repo):
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
        body="Extra context here",
    )
    with kb.connect() as conn:
        parent = kb.get_task(conn, result["parent_id"])
        assert "Extra context here" in parent.body
        assert "nousmain" in parent.body
        assert "feat/auth" in parent.body


def test_create_review_graph_parent_body_has_review_only_contract(kanban_home, fake_repo):
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    with kb.connect() as conn:
        parent = kb.get_task(conn, result["parent_id"])
        assert "REVIEW-ONLY v1" in parent.body
        assert "Do NOT modify source code" in parent.body


def test_create_review_graph_reviewer_bodies_have_review_only_contract(kanban_home, fake_repo):
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    with kb.connect() as conn:
        for rid in result["reviewer_ids"]:
            task = kb.get_task(conn, rid)
            assert "REVIEW-ONLY v1" in task.body
            assert "Do NOT modify source code" in task.body


def test_create_review_graph_base_head_in_synthesis_body(kanban_home, fake_repo):
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    with kb.connect() as conn:
        synth = kb.get_task(conn, result["synthesis_id"])
        assert "nousmain" in synth.body
        assert "feat/auth" in synth.body


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

def test_cli_review_create_json(kanban_home, fake_repo):
    from hermes_cli import kanban as kc

    out = kc.run_slash(
        f"review create 'Review PR #42' --base nousmain --head feat/auth --repo {fake_repo} --json"
    )
    payload = json.loads(out)
    assert payload["parent_id"].startswith("t_")
    assert len(payload["reviewer_ids"]) == 3
    assert payload["synthesis_id"].startswith("t_")
    assert payload["created"] is True


def test_cli_review_create_human_output(kanban_home, fake_repo):
    from hermes_cli import kanban as kc

    out = kc.run_slash(
        f"review create 'Review PR #42' --base nousmain --head feat/auth --repo {fake_repo}"
    )
    assert "Created review graph" in out
    assert "parent:" in out
    assert "reviewer 1:" in out
    assert "reviewer 2:" in out
    assert "reviewer 3:" in out
    assert "synthesis:" in out


def test_cli_review_create_idempotent_human_output(kanban_home, fake_repo):
    from hermes_cli import kanban as kc

    kc.run_slash(
        f"review create 'Review PR #42' --base nousmain --head feat/auth --repo {fake_repo}"
    )
    out = kc.run_slash(
        f"review create 'Review PR #42' --base nousmain --head feat/auth --repo {fake_repo}"
    )
    assert "Found existing review graph" in out
    assert "all cards already existed" in out


def test_cli_review_create_missing_repo(kanban_home):
    from hermes_cli import kanban as kc

    out = kc.run_slash(
        "review create 'Review PR #42' --base nousmain --head feat/auth --repo /nonexistent/path"
    )
    assert "is not a directory" in out


def test_cli_review_create_ready_flag(kanban_home, fake_repo):
    from hermes_cli import kanban as kc

    out = kc.run_slash(
        f"review create 'Review PR #42' --base nousmain --head feat/auth --repo {fake_repo} --ready --json"
    )
    payload = json.loads(out)
    with kb.connect() as conn:
        parent = kb.get_task(conn, payload["parent_id"])
        assert parent.status == "ready"


def test_cli_review_create_with_skills(kanban_home, fake_repo):
    from hermes_cli import kanban as kc

    out = kc.run_slash(
        f"review create 'Review PR #42' --base nousmain --head feat/auth --repo {fake_repo} "
        f"--skill github-code-review --skill security-scan --json"
    )
    payload = json.loads(out)
    with kb.connect() as conn:
        task = kb.get_task(conn, payload["parent_id"])
        assert "github-code-review" in task.skills
        assert "security-scan" in task.skills


def test_cli_review_create_base_head_required(kanban_home, fake_repo):
    """Missing --base or --head should produce a usage error."""
    from hermes_cli import kanban as kc

    out = kc.run_slash(
        f"review create 'Review PR #42' --repo {fake_repo}"
    )
    assert "usage error" in out.lower()


# ---------------------------------------------------------------------------
# Hardened template contract tests
# ---------------------------------------------------------------------------

def test_reviewer_bodies_contain_exact_diff_command(kanban_home, fake_repo):
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    with kb.connect() as conn:
        for rid in result["reviewer_ids"]:
            task = kb.get_task(conn, rid)
            assert "git diff nousmain...feat/auth --stat" in task.body
            assert "git diff nousmain...feat/auth\n" in task.body


def test_reviewer_bodies_contain_severity_labels(kanban_home, fake_repo):
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    with kb.connect() as conn:
        for rid in result["reviewer_ids"]:
            task = kb.get_task(conn, rid)
            assert "**Critical**" in task.body
            assert "**Important**" in task.body
            assert "**Optional/Nit**" in task.body


def test_reviewer_bodies_contain_kanban_contract(kanban_home, fake_repo):
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    with kb.connect() as conn:
        for rid in result["reviewer_ids"]:
            task = kb.get_task(conn, rid)
            assert "kanban_complete" in task.body
            assert "kanban_block" in task.body
            assert '"findings":' in task.body


def test_reviewer_bodies_contain_role_specific_checklists(kanban_home, fake_repo):
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    expected = {
        "code-quality": [
            "Readability and naming conventions",
            "DRY violations",
            "Complexity and function length",
            "Architectural consistency",
            "Type safety",
            "Documentation completeness",
        ],
        "security": [
            "Injection vectors",
            "Hardcoded secrets",
            "Input validation",
            "Authentication/authorization gaps",
            "Unsafe deserialization",
            "Dependency risks",
            "Privilege escalation",
        ],
        "test-coverage": [
            "New logic has accompanying tests",
            "Edge cases are covered",
            "Regression tests",
            "Test readability",
            "CI pass status",
            "Integration / E2E coverage",
        ],
    }
    with kb.connect() as conn:
        roles = ["code-quality", "security", "test-coverage"]
        for rid, role in zip(result["reviewer_ids"], roles):
            task = kb.get_task(conn, rid)
            for snippet in expected[role]:
                assert snippet in task.body, f"{role} body missing: {snippet}"


def test_synthesis_body_contains_go_no_go(kanban_home, fake_repo):
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    with kb.connect() as conn:
        synth = kb.get_task(conn, result["synthesis_id"])
        assert "GO/NO-GO decision" in synth.body


def test_synthesis_body_contains_all_required_sections(kanban_home, fake_repo):
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    with kb.connect() as conn:
        synth = kb.get_task(conn, result["synthesis_id"])
        assert "Blockers" in synth.body
        assert "Recommended fixes" in synth.body
        assert "Acknowledged risks" in synth.body
        assert "Rollback plan" in synth.body
        assert "Evidence reviewed" in synth.body


def test_synthesis_body_contains_default_no_go_on_critical(kanban_home, fake_repo):
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    with kb.connect() as conn:
        synth = kb.get_task(conn, result["synthesis_id"])
        assert "NO-GO" in synth.body
        assert "Critical finding exists" in synth.body


def test_synthesis_body_contains_kanban_contract(kanban_home, fake_repo):
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    with kb.connect() as conn:
        synth = kb.get_task(conn, result["synthesis_id"])
        assert "kanban_complete" in synth.body
        assert "kanban_block" in synth.body
        assert '"ship_decision":' in synth.body
        assert '"blockers":' in synth.body
        assert '"recommended_fixes":' in synth.body
        assert '"acknowledged_risks":' in synth.body
        assert '"rollback_plan":' in synth.body
        assert '"evidence_reviewed":' in synth.body


def test_generated_bodies_do_not_reference_skills(kanban_home, fake_repo):
    """Template bodies must not instruct workers to load skills that may be
    missing from the Miki profile.
    """
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    with kb.connect() as conn:
        for tid in result["reviewer_ids"] + [result["synthesis_id"]]:
            task = kb.get_task(conn, tid)
            # Reject explicit skill-loading instructions (case-insensitive)
            lower = task.body.lower()
            assert "load the `" not in lower
            assert "use the `" not in lower
            assert "skill `" not in lower


# ---------------------------------------------------------------------------
# Synthesis promotion tests
# ---------------------------------------------------------------------------

def test_reviewer_completion_promotes_synthesis(kanban_home, fake_repo):
    """When all three reviewers are marked done, complete_task's internal
    recompute_ready promotes the synthesis card from todo to ready."""
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
        ready=True,
    )
    with kb.connect() as conn:
        synth = kb.get_task(conn, result["synthesis_id"])
        assert synth.status == "todo"

        for rid in result["reviewer_ids"]:
            kb.complete_task(conn, rid, summary="review done")

        # complete_task calls recompute_ready internally; the third completion
        # promotes the synthesis automatically.
        synth = kb.get_task(conn, result["synthesis_id"])
        assert synth.status == "ready"


def test_synthesis_stays_todo_until_all_reviewers_done(kanban_home, fake_repo):
    """If only two of three reviewers are done, synthesis stays in todo."""
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
        ready=True,
    )
    with kb.connect() as conn:
        for rid in result["reviewer_ids"][:2]:
            kb.complete_task(conn, rid, summary="review done")

        synth = kb.get_task(conn, result["synthesis_id"])
        assert synth.status == "todo"


# ---------------------------------------------------------------------------
# Self-contained body tests
# ---------------------------------------------------------------------------

def test_self_contained_reviewer_bodies(kanban_home, fake_repo):
    """Reviewer bodies must contain everything the worker needs without
    relying on conversation history or external context.
    """
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    ws_path = str(Path(fake_repo).resolve())
    with kb.connect() as conn:
        for rid in result["reviewer_ids"]:
            task = kb.get_task(conn, rid)
            body = task.body
            assert "nousmain" in body
            assert "feat/auth" in body
            assert ws_path in body
            assert "git diff" in body
            assert "REVIEW-ONLY" in body
            assert "kanban_complete" in body
            assert "kanban_block" in body
            assert "**Critical**" in body
            assert "Checklist:" in body


def test_self_contained_synthesis_body(kanban_home, fake_repo):
    """Synthesis body must contain everything the worker needs without
    relying on conversation history or external context.
    """
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    ws_path = str(Path(fake_repo).resolve())
    with kb.connect() as conn:
        synth = kb.get_task(conn, result["synthesis_id"])
        body = synth.body
        assert "nousmain" in body
        assert "feat/auth" in body
        assert ws_path in body
        assert "code-quality" in body
        assert "security" in body
        assert "test-coverage" in body
        assert "GO/NO-GO" in body
        assert "kanban_complete" in body
        assert "kanban_block" in body


def test_self_contained_parent_body(kanban_home, fake_repo):
    """Parent body must contain everything the worker needs without
    relying on conversation history or external context.
    """
    result = kr.create_review_graph(
        title="Review PR #42",
        base="nousmain",
        head="feat/auth",
        repo_path=fake_repo,
    )
    ws_path = str(Path(fake_repo).resolve())
    with kb.connect() as conn:
        parent = kb.get_task(conn, result["parent_id"])
        body = parent.body
        assert "nousmain" in body
        assert "feat/auth" in body
        assert ws_path in body
        assert "REVIEW-ONLY" in body
        assert "code-quality" in body
        assert "security" in body
        assert "test-coverage" in body
        assert "Synthesis card will aggregate" in body


# ---------------------------------------------------------------------------
# JSON CLI output structure
# ---------------------------------------------------------------------------

def test_cli_review_create_json_structure(kanban_home, fake_repo):
    """JSON output must contain exact keys with correct types."""
    from hermes_cli import kanban as kc

    out = kc.run_slash(
        f"review create 'Review PR #42' --base nousmain --head feat/auth --repo {fake_repo} --json"
    )
    payload = json.loads(out)
    assert set(payload.keys()) == {"parent_id", "reviewer_ids", "synthesis_id", "created"}
    assert isinstance(payload["parent_id"], str)
    assert isinstance(payload["reviewer_ids"], list)
    assert len(payload["reviewer_ids"]) == 3
    for rid in payload["reviewer_ids"]:
        assert isinstance(rid, str)
        assert rid.startswith("t_")
    assert isinstance(payload["synthesis_id"], str)
    assert payload["synthesis_id"].startswith("t_")
    assert isinstance(payload["created"], bool)


def test_cli_review_create_json_idempotent_returns_false(kanban_home, fake_repo):
    """Second invocation with same params must return created=False."""
    from hermes_cli import kanban as kc

    kc.run_slash(
        f"review create 'Review PR #42' --base nousmain --head feat/auth --repo {fake_repo} --json"
    )
    out = kc.run_slash(
        f"review create 'Review PR #42' --base nousmain --head feat/auth --repo {fake_repo} --json"
    )
    payload = json.loads(out)
    assert payload["created"] is False

