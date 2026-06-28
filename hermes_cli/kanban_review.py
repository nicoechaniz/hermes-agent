"""Kanban ship-review graph creation.

Provides ``create_review_graph()`` — a helper that builds a durable
5-card review graph for a git change:

  1. Parent review card (base..head change summary)
  2. Code-quality reviewer   ┐
  3. Security reviewer       │  parallel
  4. Test-coverage reviewer  ┘
  5. Synthesis card          ←  gated on 2-4

All cards use deterministic idempotency keys so repeated invocations are
idempotent.  By default every card is created in ``triage`` so nothing
dispatches until the operator explicitly promotes them.

The CLI surface lives in ``hermes_cli/kanban.py`` under
``hermes kanban review create …``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Typed spec
# ---------------------------------------------------------------------------

@dataclass
class ReviewGraphSpec:
    """Parameters that fully describe a ship-review graph."""

    repo_path: str
    base: str
    head: str
    title: str
    assignee: Optional[str] = None
    ready: bool = False
    idempotency_prefix: Optional[str] = None
    skills: list[str] = field(default_factory=list)
    body: Optional[str] = None


# ---------------------------------------------------------------------------
# Idempotency helpers
# ---------------------------------------------------------------------------

def _repo_hash(repo_path: str) -> str:
    """Stable 16-char hex hash of the resolved repo path."""
    abs_path = str(Path(repo_path).resolve())
    return hashlib.sha256(abs_path.encode()).hexdigest()[:16]


def _review_base_key(base: str, head: str, repo_path: str) -> str:
    """Deterministic base key for a given review target.

    Derives from repo realpath hash + base + head so the key is stable
    across board switches and path aliasing.
    """
    return f"ship-review:{_repo_hash(repo_path)}:{base}:{head}"


def _card_key(base_key: str, role: str) -> str:
    return f"{base_key}:{role}"


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

_ROLE_FOCUS = {
    "code-quality": (
        "readability, maintainability, naming, complexity, DRY violations, "
        "and architectural consistency"
    ),
    "security": (
        "injection vectors, unsafe evals, hardcoded secrets, input validation, "
        "auth/authz gaps, and dependency risks"
    ),
    "test-coverage": (
        "missing tests for new logic, edge cases, regression tests, "
        "test readability, and CI pass status"
    ),
}

_ROLE_CHECKLIST: dict[str, list[str]] = {
    "code-quality": [
        "Readability and naming conventions",
        "DRY violations and duplicated logic",
        "Complexity and function length",
        "Architectural consistency with existing patterns",
        "Type safety and static analysis concerns",
        "Documentation completeness",
    ],
    "security": [
        "Injection vectors (SQL, command, path, eval)",
        "Hardcoded secrets or credentials",
        "Input validation and sanitization",
        "Authentication/authorization gaps",
        "Unsafe deserialization or eval usage",
        "Dependency risks (untrusted sources, version pinning)",
        "Privilege escalation paths",
    ],
    "test-coverage": [
        "New logic has accompanying tests",
        "Edge cases are covered",
        "Regression tests for bug fixes",
        "Test readability and naming",
        "CI pass status and flaky test checks",
        "Integration / E2E coverage for user-facing changes",
    ],
}


def _reviewer_body(role: str, base: str, head: str, ws_path: str) -> str:
    """Return a hardened reviewer task body for *role*."""
    focus = _ROLE_FOCUS.get(role, "general code review")
    checklist_lines = "\n".join(f"- [ ] {item}" for item in _ROLE_CHECKLIST.get(role, []))

    return (
        f"Review the {role} of `{base}` → `{head}` in `{ws_path}`.\n\n"
        f"Diff to review:\n"
        f"  git diff {base}...{head} --stat\n"
        f"  git diff {base}...{head}\n\n"
        f"**REVIEW-ONLY v1** — Do NOT modify source code. "
        f"Report findings as structured metadata only.\n\n"
        f"Severity labels:\n"
        f"- **Critical** — Merge blocker; must be fixed before ship.\n"
        f"- **Important** — Significant concern; strongly recommend fixing.\n"
        f"- **Optional/Nit** — Minor improvement; ship at discretion.\n\n"
        f"kanban_complete / kanban_block contract:\n"
        f'- Call kanban_complete(summary=..., metadata={{"findings": [...]}})\n'
        f'- Call kanban_block(reason=...) if you are blocked '
        f"(missing context, cannot access files)\n"
        f"- Each finding must include: severity, file, line (if applicable), "
        f"issue description.\n\n"
        f"Focus on: {focus}.\n\n"
        f"Checklist:\n{checklist_lines}"
    )


def _synthesis_body(base: str, head: str, ws_path: str) -> str:
    """Return a hardened synthesis task body."""
    return (
        f"Synthesize findings from the three reviewers for `{base}` → `{head}` "
        f"in `{ws_path}`.\n\n"
        f"Inputs: parent card + three completed reviewer cards "
        f"(code-quality, security, test-coverage).\n\n"
        f"Required output structure:\n"
        f"- **GO/NO-GO decision** with explicit rationale.\n"
        f"- **Blockers**: list of Critical findings that must be resolved before ship.\n"
        f"- **Recommended fixes**: ordered by priority (Critical first, then Important).\n"
        f"- **Acknowledged risks**: Important/Optional findings accepted as-is with justification.\n"
        f"- **Rollback plan**: steps to revert this change if issues surface in production.\n"
        f"- **Evidence reviewed**: list of files/evidence examined "
        f"(diff stat, key changed files).\n\n"
        f"Default rule: **NO-GO** if any Critical finding exists unless the user "
        f"explicitly accepts the risk in writing.\n\n"
        f"kanban_complete / kanban_block contract:\n"
        f'- Call kanban_complete(summary=..., metadata={{"ship_decision": "GO|NO-GO", '
        f'"blockers": [...], "recommended_fixes": [...], '
        f'"acknowledged_risks": [...], "rollback_plan": "...", '
        f'"evidence_reviewed": [...]}})\n'
        f'- Call kanban_block(reason=...) if you are blocked '
        f"(missing reviewer output, incomplete context)."
    )


# ---------------------------------------------------------------------------
# Graph creation
# ---------------------------------------------------------------------------

def create_review_graph(
    *,
    title: str,
    base: str,
    head: str,
    repo_path: str,
    board: Optional[str] = None,
    assignee: Optional[str] = None,
    ready: bool = False,
    body: Optional[str] = None,
    skills: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Create (or return existing) ship-review graph.

    Parameters
    ----------
    title:
        Human title for the parent review card (e.g. "Review PR #42").
    base:
        Git base ref (e.g. ``nousmain``).
    head:
        Git head ref (e.g. ``feat/auth``).
    repo_path:
        Absolute path to the repository root. Used as ``dir:`` workspace.
    board:
        Board slug. Defaults to ``kanban_db.get_current_board()``.
    assignee:
        Profile name for **all** cards. ``None`` leaves them unassigned.
    ready:
        When ``False`` (default) every card is created in ``triage``.
        When ``True`` the parent + reviewer cards are created in ``ready``
        and the synthesis card in ``todo`` (it will auto-promote once its
        parents complete).
    body:
        Optional extra context appended to the parent review card body.
    skills:
        Optional list of skills to attach to every card.

    Returns
    -------
    dict with ``parent_id``, ``reviewer_ids``, ``synthesis_id``, and
    ``created`` (bool — ``False`` when every id already existed).
    """
    board = board or kb.get_current_board()
    base_key = _review_base_key(base, head, repo_path)
    ws_kind, ws_path = "dir", str(Path(repo_path).resolve())
    default_status = "ready" if ready else "triage"

    # Parent card body
    parent_body_parts = [
        f"Ship review for `{base}` → `{head}`.",
        f"Repository: {ws_path}",
        "",
        "**REVIEW-ONLY v1** — Do NOT modify source code. "
        "Report findings as structured metadata only.",
        "",
        "Reviewers:",
        "- code-quality",
        "- security",
        "- test-coverage",
        "",
        "Synthesis card will aggregate findings once all reviewers finish.",
    ]
    if body:
        parent_body_parts.extend(["", "Context:", body])
    parent_body = "\n".join(parent_body_parts)

    # Reviewer templates
    reviewers = [
        ("code-quality", f"[REVIEW] Code quality — {title}"),
        ("security", f"[REVIEW] Security — {title}"),
        ("test-coverage", f"[REVIEW] Test coverage — {title}"),
    ]

    created_any = False
    reviewer_ids: list[str] = []

    with kb.connect(board=board) as conn:
        # --- Parent review card ---
        parent_key = _card_key(base_key, "parent")
        existing_parent = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived'",
            (parent_key,),
        ).fetchone()
        if existing_parent:
            parent_id = existing_parent["id"]
        else:
            parent_id = kb.create_task(
                conn,
                title=title,
                body=parent_body,
                assignee=assignee,
                created_by=_profile_author(),
                workspace_kind=ws_kind,
                workspace_path=ws_path,
                triage=not ready,
                idempotency_key=parent_key,
                skills=skills,
            )
            created_any = True

        # --- Reviewer cards (parallel) ---
        # Note: reviewers are NOT linked to the parent card because
        # kanban_db treats every parent link as a blocking dependency.
        # The parent is an organisational umbrella; only the synthesis
        # card is gated on the reviewers.
        for role, rtitle in reviewers:
            rkey = _card_key(base_key, role)
            existing = conn.execute(
                "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived'",
                (rkey,),
            ).fetchone()
            if existing:
                rid = existing["id"]
            else:
                rid = kb.create_task(
                    conn,
                    title=rtitle,
                    body=_reviewer_body(role, base, head, ws_path),
                    assignee=assignee,
                    created_by=_profile_author(),
                    workspace_kind=ws_kind,
                    workspace_path=ws_path,
                    triage=not ready,
                    idempotency_key=rkey,
                    skills=skills,
                )
                created_any = True
            reviewer_ids.append(rid)

        # --- Synthesis card (gated on all reviewers) ---
        synthesis_body = _synthesis_body(base, head, ws_path)
        synth_key = _card_key(base_key, "synthesis")
        existing_synth = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived'",
            (synth_key,),
        ).fetchone()
        if existing_synth:
            synthesis_id = existing_synth["id"]
        else:
            synthesis_id = kb.create_task(
                conn,
                title=f"[SYNTHESIS] {title}",
                body=synthesis_body,
                assignee=assignee,
                created_by=_profile_author(),
                workspace_kind=ws_kind,
                workspace_path=ws_path,
                triage=not ready,
                idempotency_key=synth_key,
                parents=tuple(reviewer_ids),
                skills=skills,
            )
            created_any = True

    return {
        "parent_id": parent_id,
        "reviewer_ids": reviewer_ids,
        "synthesis_id": synthesis_id,
        "created": created_any,
    }


def _profile_author() -> str:
    for env in ("HERMES_PROFILE_NAME", "HERMES_PROFILE"):
        v = os.environ.get(env)
        if v:
            return v
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "user"
    except Exception:
        return "user"
