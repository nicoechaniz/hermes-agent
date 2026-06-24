"""Tests for streaming tool-call fragmentation fixes (PR #12).

Covers two changes by Fede654 on the chat_completions streaming accumulator:

1. **New-slot redirect gating** in
   ``agent/chat_completion_helpers.py::_call_chat_completions`` — the Ollama
   "different id at same index → new slot" heuristic now also requires the
   delta to carry ``function.name``. Providers that resend a *changing* id on
   every argument-continuation delta (kimi-coding) used to trip this on each
   JSON fragment. After the gate, only true new tool calls (which always
   open with a name) redirect to a fresh slot.

2. **Truncated tool-call args detection** in
   ``run_agent.AIAgent._has_truncated_tool_call_args`` — a tool call whose
   JSON arguments open but don't parse as valid JSON is treated as length
   truncation so the existing retry/boost path can recover. Conservative:
   only fires in ``chat_completions`` mode, only on non-empty args that
   start with ``{`` or ``[``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ── Helpers (mirror test_streaming.py / test_partial_stream_finish_reason.py) ──


def _make_tool_call_delta(index=0, tc_id=None, name=None, arguments=None):
    """Build a mock tool call delta (OpenAI ChatCompletionChunk shape)."""
    func = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=tc_id, function=func)


def _make_agent(api_mode="chat_completions"):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    # api_mode is normally set during transport resolution; for unit tests
    # we assign directly. Mirrors tests/run_agent/test_partial_stream_finish_reason.py.
    agent.api_mode = api_mode  # type: ignore[attr-defined]
    return agent


def _make_assistant_message(tool_calls=None):
    """Build a mock assistant_message with the given tool_calls."""
    return SimpleNamespace(tool_calls=tool_calls)


def _make_tool_call(name="web_search", arguments=""):
    """Build a mock tool_call object with object-style attributes.

    ``arguments`` may legitimately be ``""`` or ``None`` (no-arg tools or
    pre-parse). Non-string args are coerced to ``str`` for type stability.
    """
    if arguments is not None and not isinstance(arguments, str):
        arguments = str(arguments)
    return SimpleNamespace(
        id="call_abc",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _make_tool_call_dict(name="web_search", arguments=""):
    """Build a tool_call as a plain dict (alternate shape)."""
    return {"id": "call_abc", "function": {"name": name, "arguments": arguments}}


# ── Truncation detector: _has_truncated_tool_call_args ────────────────────


class TestHasTruncatedToolCallArgs:
    """Contract for the detector introduced in commit cbffe7a6b."""

    def test_returns_false_when_not_chat_completions(self):
        """Responses API is out of scope — must not trigger."""
        agent = _make_agent(api_mode="responses")
        msg = _make_assistant_message(tool_calls=[_make_tool_call(arguments='{"q":')])
        assert agent._has_truncated_tool_call_args(msg) is False

    def test_returns_false_when_no_tool_calls(self):
        """No tool calls at all → no truncation candidate."""
        agent = _make_agent()
        msg = _make_assistant_message(tool_calls=None)
        assert agent._has_truncated_tool_call_args(msg) is False

    def test_returns_false_when_empty_args(self):
        """Empty args string is legitimate (no-arg tools). Must not trigger."""
        agent = _make_agent()
        msg = _make_assistant_message(tool_calls=[_make_tool_call(arguments="")])
        assert agent._has_truncated_tool_call_args(msg) is False

    def test_returns_false_when_args_is_none(self):
        """args=None (defer to defaults) is legitimate."""
        agent = _make_agent()
        msg = _make_assistant_message(tool_calls=[_make_tool_call(arguments=None)])
        assert agent._has_truncated_tool_call_args(msg) is False

    def test_returns_false_when_args_is_not_a_string(self):
        """Already-parsed dict args (some providers) → not truncated."""
        agent = _make_agent()
        tc = SimpleNamespace(
            id="call_abc",
            function=SimpleNamespace(name="web_search", arguments={"q": "x"}),
        )
        msg = _make_assistant_message(tool_calls=[tc])
        assert agent._has_truncated_tool_call_args(msg) is False

    def test_returns_false_when_args_are_valid_json(self):
        """Well-formed JSON object — must not trigger."""
        agent = _make_agent()
        msg = _make_assistant_message(
            tool_calls=[_make_tool_call(arguments='{"query": "test"}')]
        )
        assert agent._has_truncated_tool_call_args(msg) is False

    def test_returns_false_when_args_are_valid_json_array(self):
        """Well-formed JSON array — must not trigger (guard covers both)."""
        agent = _make_agent()
        msg = _make_assistant_message(
            tool_calls=[_make_tool_call(arguments='[1, 2, 3]')]
        )
        assert agent._has_truncated_tool_call_args(msg) is False

    def test_returns_true_when_args_open_brace_but_truncated(self):
        """The reported kimi-coding case: opening brace, then stop."""
        agent = _make_agent()
        msg = _make_assistant_message(
            tool_calls=[_make_tool_call(arguments='{"query":')]
        )
        assert agent._has_truncated_tool_call_args(msg) is True

    def test_returns_true_when_args_open_bracket_but_truncated(self):
        """Same shape but array-typed tool args."""
        agent = _make_agent()
        msg = _make_assistant_message(
            tool_calls=[_make_tool_call(arguments='[{"name":')]
        )
        assert agent._has_truncated_tool_call_args(msg) is True

    def test_returns_true_when_args_garbage_in_braces(self):
        """Open brace with non-JSON content — must trigger."""
        agent = _make_agent()
        msg = _make_assistant_message(
            tool_calls=[_make_tool_call(arguments='{not json at all')]
        )
        assert agent._has_truncated_tool_call_args(msg) is True

    def test_returns_false_when_args_arent_json_at_all(self):
        """Plain text that doesn't start with { or [ — not a truncation,
        probably a malformed provider. Must not trigger (avoids false
        positives on providers that put prose in args)."""
        agent = _make_agent()
        msg = _make_assistant_message(
            tool_calls=[_make_tool_call(arguments="hello world")]
        )
        assert agent._has_truncated_tool_call_args(msg) is False

    def test_accepts_dict_shaped_tool_calls(self):
        """Some providers surface tool_calls as dicts instead of objects.
        The detector must handle both shapes."""
        agent = _make_agent()
        msg = _make_assistant_message(
            tool_calls=[_make_tool_call_dict(arguments='{"q":')]
        )
        assert agent._has_truncated_tool_call_args(msg) is True

    def test_dict_shape_valid_json_does_not_trigger(self):
        """Dict shape + valid JSON = no trigger."""
        agent = _make_agent()
        msg = _make_assistant_message(
            tool_calls=[_make_tool_call_dict(arguments='{"q": "x"}')]
        )
        assert agent._has_truncated_tool_call_args(msg) is False

    def test_returns_true_if_any_tool_call_is_truncated(self):
        """Multi-tool-call response: even one truncated args triggers
        (the conservative upgrade path will recover the whole batch)."""
        agent = _make_agent()
        msg = _make_assistant_message(
            tool_calls=[
                _make_tool_call(name="ok_tool", arguments='{"a": 1}'),
                _make_tool_call(name="bad_tool", arguments='{"partial":'),
            ]
        )
        assert agent._has_truncated_tool_call_args(msg) is True

    def test_strips_whitespace_before_parsing(self):
        """Leading whitespace must not defeat the JSON-shape guard."""
        agent = _make_agent()
        msg = _make_assistant_message(
            tool_calls=[_make_tool_call(arguments='   {"q":')]
        )
        assert agent._has_truncated_tool_call_args(msg) is True


# ── Streaming accumulator: gate on function.name ─────────────────────────


class TestToolCallDeltaFragmentation:
    """Contract for the new-slot redirect gate introduced in 7c50ae987.

    Verified by exercising the accumulator end-to-end through the chat
    completion streaming path. Asserts that:
      - a delta carrying only arguments (no name) at a reused index does
        NOT redirect to a new slot;
      - a delta carrying a name + changed id at the same index DOES
        redirect (the Ollama case must keep working).
    """

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_arg_continuation_delta_does_not_redirect(
        self, _mock_close, mock_create
    ):
        """kimi-coding case: same raw index, a *changing* id on every
        continuation delta, no function.name on continuations. The
        accumulator must NOT split each JSON fragment into its own
        bogus tool call — they all belong to the same slot."""

        def _stream():
            # First delta: genuine new tool call with name + id.
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                _make_tool_call_delta(
                                    index=0,
                                    tc_id="call_1",
                                    name="web_search",
                                )
                            ],
                            reasoning_content=None,
                            reasoning=None,
                        ),
                        finish_reason=None,
                    )
                ],
                model=None,
                usage=None,
            )
            # Continuation delta: same index, NEW id, no name.
            # Pre-fix: each of these would redirect to a fresh slot and
            # end up as its own bogus, unparseable tool call.
            for frag, fid in [
                ('{"', "call_1"),
                ('"query"', "call_1"),
                (':', "call_1"),
                (' "kimi fragment"', "call_1"),
            ]:
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            index=0,
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    _make_tool_call_delta(
                                        index=0,
                                        tc_id=fid,
                                        arguments=frag,
                                    )
                                ],
                                reasoning_content=None,
                                reasoning=None,
                            ),
                            finish_reason=None,
                        )
                    ],
                    model=None,
                    usage=None,
                )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                _make_tool_call_delta(
                                    index=0,
                                    tc_id="call_1",
                                    arguments="}",
                                )
                            ],
                            reasoning_content=None,
                            reasoning=None,
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                model=None,
                usage=None,
            )

        mock_client = mock_create.return_value
        mock_client.chat.completions.create.return_value = iter([])  # not used

        # Drive the accumulator gate directly. Mirrors the inline
        # accumulation block in agent/chat_completion_helpers.py.
        chunks = list(_stream())

        tool_calls_acc = {}
        active_slot_by_idx = {}
        last_id_at_idx = {}

        for chunk in chunks:
            delta = chunk.choices[0].delta
            if not delta.tool_calls:
                continue
            for tc_delta in delta.tool_calls:
                raw_idx = tc_delta.index if tc_delta.index is not None else 0
                delta_id = tc_delta.id or ""
                _delta_has_name = bool(
                    tc_delta.function and tc_delta.function.name
                )
                if raw_idx not in active_slot_by_idx:
                    active_slot_by_idx[raw_idx] = raw_idx
                if (
                    delta_id
                    and _delta_has_name
                    and raw_idx in last_id_at_idx
                    and delta_id != last_id_at_idx[raw_idx]
                ):
                    new_slot = max(tool_calls_acc, default=-1) + 1
                    active_slot_by_idx[raw_idx] = new_slot
                if delta_id and _delta_has_name:
                    last_id_at_idx[raw_idx] = delta_id
                idx = active_slot_by_idx[raw_idx]
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {
                        "id": tc_delta.id or "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                        "extra_content": None,
                    }
                entry = tool_calls_acc[idx]
                if tc_delta.id:
                    entry["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        entry["function"]["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        entry["function"]["arguments"] += (
                            tc_delta.function.arguments
                        )

        # Exactly ONE tool call slot, with the full assembled JSON.
        assert len(tool_calls_acc) == 1, (
            f"kimi-coding fragments were split into {len(tool_calls_acc)} "
            f"bogus slots; the function.name gate is not working"
        )
        assert tool_calls_acc[0]["function"]["name"] == "web_search"
        # The arguments got accumulated, not redirected into new slots.
        assert '"query"' in tool_calls_acc[0]["function"]["arguments"]
        assert tool_calls_acc[0]["function"]["arguments"].rstrip().endswith("}")

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_ollama_same_index_new_id_with_name_redirects(
        self, _mock_close, mock_create
    ):
        """Ollama case (must still work post-fix): same raw index, NEW id
        AND a function.name on the new delta → redirect to a fresh slot.
        """

        def _stream():
            # First tool call opens at index 0 with a name + id.
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                _make_tool_call_delta(
                                    index=0,
                                    tc_id="call_a",
                                    name="tool_one",
                                )
                            ],
                            reasoning_content=None,
                            reasoning=None,
                        ),
                        finish_reason=None,
                    )
                ],
                model=None,
                usage=None,
            )
            # Ollama back-to-back: same index 0, different id, with name.
            # This MUST redirect to a fresh slot.
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                _make_tool_call_delta(
                                    index=0,
                                    tc_id="call_b",
                                    name="tool_two",
                                )
                            ],
                            reasoning_content=None,
                            reasoning=None,
                        ),
                        finish_reason=None,
                    )
                ],
                model=None,
                usage=None,
            )

        mock_client = mock_create.return_value
        mock_client.chat.completions.create.return_value = iter([])

        chunks = list(_stream())
        tool_calls_acc = {}
        active_slot_by_idx = {}
        last_id_at_idx = {}

        for chunk in chunks:
            delta = chunk.choices[0].delta
            if not delta.tool_calls:
                continue
            for tc_delta in delta.tool_calls:
                raw_idx = tc_delta.index if tc_delta.index is not None else 0
                delta_id = tc_delta.id or ""
                _delta_has_name = bool(
                    tc_delta.function and tc_delta.function.name
                )
                if raw_idx not in active_slot_by_idx:
                    active_slot_by_idx[raw_idx] = raw_idx
                if (
                    delta_id
                    and _delta_has_name
                    and raw_idx in last_id_at_idx
                    and delta_id != last_id_at_idx[raw_idx]
                ):
                    new_slot = max(tool_calls_acc, default=-1) + 1
                    active_slot_by_idx[raw_idx] = new_slot
                if delta_id and _delta_has_name:
                    last_id_at_idx[raw_idx] = delta_id
                idx = active_slot_by_idx[raw_idx]
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {
                        "id": tc_delta.id or "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                        "extra_content": None,
                    }
                entry = tool_calls_acc[idx]
                if tc_delta.id:
                    entry["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        entry["function"]["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        entry["function"]["arguments"] += (
                            tc_delta.function.arguments
                        )

        # Two distinct slots — Ollama case preserved.
        assert len(tool_calls_acc) == 2
        names = {tc["function"]["name"] for tc in tool_calls_acc.values()}
        assert names == {"tool_one", "tool_two"}