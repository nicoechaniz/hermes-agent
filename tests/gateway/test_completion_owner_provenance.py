"""A process's inherited transport is not authority to promote its worker.

Exercise producer -> gateway injection -> native receiver with real SQLite and
SessionStore. Stop immediately after session resolution, before any model call.
"""
import asyncio
from collections import OrderedDict
from types import SimpleNamespace
import threading

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from hermes_state import AsyncSessionDB
from hermes_cli import goals


class _ResolvedBoundary(Exception):
    pass


@pytest.fixture
def native_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig()
    store = SessionStore(tmp_path / "sessions", config)
    db = store._db
    assert db is not None
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="123", chat_type="dm",
        user_id="123", thread_id="456",
    )
    parent = store.get_or_create_session(source)
    db.append_message(parent.session_id, role="user", content="Continue the human mission")
    goals.save_goal(parent.session_id, goals.GoalState(goal="human mission"))
    db.create_session(
        "worker", source="subagent", parent_session_id=parent.session_id,
        model_config={"_delegate_from": parent.session_id},
    )
    db.append_message("worker", role="user", content="Only inspect the worker task")
    goals.save_goal("worker", goals.GoalState(goal="worker task"))

    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner._running = True
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_db = AsyncSessionDB(db)
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    monkeypatch.setattr(runner, "_recover_telegram_topic_thread_id", lambda _source: None)
    resolved = []
    events = []

    def stop_before_model(key, _source):
        resolved.append(store.lookup_by_session_key(key).session_id)
        raise _ResolvedBoundary

    monkeypatch.setattr(runner, "_cache_session_source", stop_before_model)

    async def handle_message(event):
        events.append(event)
        try:
            await runner._handle_message_with_agent(event, event.source, parent.session_key, 1)
        except _ResolvedBoundary:
            pass

    runner.adapters = {Platform.TELEGRAM: SimpleNamespace(handle_message=handle_message)}
    boundary = SimpleNamespace(
        runner=runner, store=store, db=db, source=source, parent=parent,
        parent_id=parent.session_id,
        resolved=resolved, events=events,
    )
    yield boundary
    db.close()


def _completion(boundary, owner="worker"):
    return {
        "type": "completion", "session_id": "proc_worker", "started_at": 1234.5,
        "session_key": boundary.parent.session_key, "parent_session_id": owner,
        "platform": "telegram", "chat_id": "123", "thread_id": "456", "chat_type": "dm",
        "command": "test command", "exit_code": 0, "output": "finished",
    }


def _assert_foreground_unchanged(b):
    assert b.db.get_session(b.parent_id)["ended_at"] is None
    current = b.store.lookup_by_session_key(b.parent.session_key)
    assert current.session_id == b.parent_id
    assert b.db.get_messages(current.session_id)[0]["content"] == "Continue the human mission"
    assert goals.load_goal(current.session_id) == goals.GoalState(goal="human mission")
    assert b.db.get_messages("worker")[0]["content"] == "Only inspect the worker task"
    assert goals.load_goal("worker") == goals.GoalState(goal="worker task")


def test_worker_process_completion_cannot_promote_worker(native_boundary):
    b = native_boundary
    result = asyncio.run(b.runner._deliver_completion_notification("process finished", _completion(b)))
    _assert_foreground_unchanged(b)
    assert result is None  # terminal non-gateway owner, not a retry or an ack
    assert b.events == []
    assert b.resolved == []


@pytest.mark.parametrize("source", ["subagent", "telegram", "gateway"])
def test_queued_worker_pin_is_rejected_at_receiver(native_boundary, source):
    """A queued event can bypass the producer gate; source is mutable metadata."""
    b = native_boundary
    b.db.record_gateway_session_peer(
        "worker", source=source, session_key=b.parent.session_key,
        chat_id="123", chat_type="dm", thread_id="456",
    )
    assert b.db.get_session("worker")["source"] == source
    event = MessageEvent(
        text="late process result", message_type=MessageType.TEXT,
        source=b.source, internal=True, metadata={"gateway_session_id": "worker"},
    )
    asyncio.run(b.runner.adapters[Platform.TELEGRAM].handle_message(event))
    _assert_foreground_unchanged(b)
    assert b.resolved == []


@pytest.mark.parametrize("boundary", ["preflight", "receiver"])
def test_source_only_worker_is_not_a_compression_delivery_tip(native_boundary, boundary):
    b = native_boundary
    b.db.end_session(b.parent_id, "compression")
    b.db.create_session("legacy-worker", source="subagent", parent_session_id=b.parent_id)
    # The legacy SQL lineage walker excludes marker-bearing delegates, but a
    # source-only worker can still be returned. The delivery boundary must check.
    assert b.db.get_compression_tip(b.parent_id) == "legacy-worker"
    if boundary == "preflight":
        assert asyncio.run(b.runner._classify_completion_target(b.parent_id)) == "terminal"
    else:
        assert asyncio.run(b.runner._resolve_async_delegation_session(b.parent, b.parent_id)) is None
        assert b.store._entries[b.parent.session_key].session_id == b.parent_id


def test_queued_completion_cannot_recover_route_into_source_only_worker(native_boundary):
    b = native_boundary
    b.db.end_session(b.parent_id, "compression")
    b.db.create_session("legacy-worker", source="subagent", parent_session_id=b.parent_id)
    event = MessageEvent(text="queued result", source=b.source, internal=True,
                         metadata={"gateway_session_id": b.parent_id})
    asyncio.run(b.runner.adapters[Platform.TELEGRAM].handle_message(event))
    assert b.resolved == []
    assert b.store.lookup_by_session_key(b.parent.session_key).session_id == b.parent_id
    assert b.db.get_session(b.parent_id)["end_reason"] == "compression"


@pytest.mark.parametrize("owner", ["parent", "worker", "grandchild"])
@pytest.mark.parametrize("count", [1, 3])
def test_native_watchers_settle_without_forwarding_workers(
    native_boundary, monkeypatch, tmp_path, caplog, owner, count,
):
    """Real watcher, batching, classifier, injector and receiver; no child process."""
    import tools.process_registry as pr

    b = native_boundary
    b.db.create_session(
        "grandchild", source="subagent", parent_session_id="worker",
        model_config={"_delegate_from": "worker"},
    )
    owner_id = b.parent_id if owner == "parent" else owner
    monkeypatch.setattr(pr, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = pr.ProcessRegistry()
    monkeypatch.setattr(pr, "process_registry", registry)
    b.runner._completion_notification_batch_window = 0
    watchers = []
    for index in range(count):
        sid = f"proc_{index}"
        registry._finished[sid] = pr.ProcessSession(
            id=sid, command="bounded fixture", task_id="default", started_at=1234.5,
            exited=True, exit_code=0, output_buffer="still queryable\n",
            parent_session_id=owner_id, notify_on_complete=True,
        )
        watchers.append({
            "session_id": sid, "check_interval": 0, "notify_on_complete": True,
            "session_key": b.parent.session_key, "platform": "telegram",
            "chat_type": "dm", "chat_id": "123", "thread_id": "456",
            # Exercise the ProcessSession owner fallback, not just watcher data.
        })

    async def run_watchers():
        await asyncio.wait_for(asyncio.gather(*(
            b.runner._run_process_watcher(w) for w in watchers
        )), timeout=3)

    asyncio.run(run_watchers())
    _assert_foreground_unchanged(b)
    assert b.resolved == ([b.parent_id] if owner == "parent" else [])
    if owner != "parent":
        assert b.events == []
        assert "delegated owner" in caplog.text
        assert b.runner._completion_deliveries_delivered == {}
    for index in range(count):
        assert registry.get(f"proc_{index}").output_buffer == "still queryable\n"


@pytest.mark.parametrize("end_reason", [None, "idle_timeout", "compression"])
def test_worker_lifecycle_never_becomes_parent_authority(native_boundary, end_reason):
    b = native_boundary
    # Real peer refresh masks the original source; _delegate_from is durable.
    b.db.record_gateway_session_peer(
        "worker", source="telegram", session_key=b.parent.session_key,
    )
    if end_reason:
        b.db.end_session("worker", end_reason)
    if end_reason == "compression":
        b.db.create_session(
            "worker-tip", source="telegram", parent_session_id="worker",
            model_config={"_delegate_from": b.parent_id},
        )
    for owner in (["worker", "worker-tip"] if end_reason == "compression" else ["worker"]):
        assert asyncio.run(b.runner._deliver_completion_notification("finished", _completion(b, owner))) is None
        assert asyncio.run(b.runner._resolve_async_delegation_session(b.parent, owner)) is None
    _assert_foreground_unchanged(b)
    assert b.events == []


def test_parent_owned_delegation_result_still_delivers(native_boundary):
    b = native_boundary
    evt = _completion(b, b.parent_id)
    evt.update(type="async_delegation", delegation_id="")
    assert asyncio.run(b.runner._deliver_completion_notification("worker summary", evt)) is True
    _assert_foreground_unchanged(b)
    assert b.resolved == [b.parent_id]


def test_nested_delegation_result_is_not_forwarded_to_foreground(native_boundary):
    b = native_boundary
    evt = _completion(b, "worker")
    evt.update(type="async_delegation", delegation_id="")
    assert asyncio.run(b.runner._deliver_completion_notification("grandchild summary", evt)) is None
    _assert_foreground_unchanged(b)
    assert b.events == []


@pytest.mark.parametrize("route", ["parent", "middle", "tip"])
def test_verified_compression_delivery_preserves_human_goal(native_boundary, route):
    b = native_boundary
    goal = b.db.get_meta(f"goal:{b.parent_id}")
    previous = b.parent_id
    for sid in ("middle", "tip"):
        b.db.end_session(previous, "compression")
        b.db.create_session(sid, source="telegram", parent_session_id=previous)
        previous = sid
    b.db.set_meta("goal:tip", goal)
    if route != "parent":
        # Native CAS route movement, not manual /resume reopening the parent.
        b.store.advance_compression_session(b.parent.session_key, b.parent_id, route)
    assert asyncio.run(b.runner._deliver_completion_notification("finished", _completion(b, b.parent_id))) is True
    assert b.resolved == ["tip"]
    assert b.store.lookup_by_session_key(b.parent.session_key).session_id == "tip"
    assert b.db.get_session(b.parent_id)["end_reason"] == "compression"
    assert b.db.get_session("tip")["ended_at"] is None
    assert b.db.get_meta("goal:tip") == goal


@pytest.mark.parametrize("reason", ["session_reset", "new_session", "user_exit", "session_switch"])
def test_user_boundary_blocks_delayed_parent_completion(native_boundary, reason):
    b = native_boundary
    b.db.end_session(b.parent_id, reason)
    assert asyncio.run(b.runner._deliver_completion_notification("stale", _completion(b, b.parent_id))) is None
    assert asyncio.run(b.runner._resolve_async_delegation_session(b.parent, b.parent_id)) is None
    assert b.db.get_session(b.parent_id)["end_reason"] == reason
    assert b.events == []


def test_compression_pin_cannot_cross_new_route(native_boundary):
    b = native_boundary
    b.db.end_session(b.parent_id, "compression")
    b.db.create_session("tip", source="telegram", parent_session_id=b.parent_id)
    b.store.advance_compression_session(b.parent.session_key, b.parent_id, "tip")
    fresh = b.store.reset_session(b.parent.session_key)
    fresh_id = fresh.session_id
    assert asyncio.run(b.runner._resolve_async_delegation_session(fresh, b.parent_id)) is None
    assert b.store.lookup_by_session_key(b.parent.session_key).session_id == fresh_id
    assert b.db.get_session(fresh_id)["ended_at"] is None


@pytest.mark.parametrize("boundary", ["receiver", "native_delivery", "tip", "preflight"])
def test_reset_of_stale_compression_route_is_not_continuation(native_boundary, boundary):
    b = native_boundary
    b.db.end_session(b.parent_id, "compression")
    b.db.create_session("tip", source="telegram", parent_session_id=b.parent_id)
    # /new may run before the routing index catches up to the compression tip.
    fresh = b.store.reset_session(b.parent.session_key)
    fresh_id = fresh.session_id
    if boundary == "tip":
        assert b.db.get_compression_tip(b.parent_id) == "tip"
    elif boundary == "preflight":
        assert asyncio.run(b.runner._classify_completion_target(b.parent_id, b.parent.session_key)) == "terminal"
    elif boundary == "receiver":
        assert asyncio.run(b.runner._resolve_async_delegation_session(fresh, b.parent_id)) is None
    else:
        asyncio.run(b.runner._deliver_completion_notification("stale", _completion(b, b.parent_id)))
        assert b.resolved == []
    assert b.store.lookup_by_session_key(b.parent.session_key).session_id == fresh_id
    assert b.db.get_session(fresh_id)["ended_at"] is None


def test_manual_resume_store_operation_is_separate_from_notification_authority(native_boundary):
    b = native_boundary
    # /resume authorization belongs to its command handler, not this storage
    # operation. A notification must not remove the manual operation itself.
    switched = b.store.switch_session(b.parent.session_key, "worker")
    assert switched.session_id == "worker"
    assert b.db.get_session(b.parent_id)["end_reason"] == "session_switch"
    assert b.db.get_session("worker")["ended_at"] is None


@pytest.mark.parametrize("existing_human_goal", [False, True])
def test_already_bound_worker_origin_can_receive_own_completion(native_boundary, monkeypatch, existing_human_goal):
    b = native_boundary
    # Give a historical worker-origin row native same-user peer provenance.
    # Then execute the real /resume handler and its ownership authorization.
    b.db.record_gateway_session_peer(
        "worker", source="telegram", user_id="123", chat_id="123",
        chat_type="dm", thread_id="456", session_key=b.parent.session_key,
    )
    for hook in ("_release_running_agent_state", "_clear_conversation_scope", "_evict_cached_agent"):
        monkeypatch.setattr(b.runner, hook, lambda *_a, **_kw: None)
    resume = MessageEvent(text="/resume worker", source=b.source)
    confirmation = asyncio.run(b.runner._handle_resume_command(resume))
    assert "resumed" in confirmation.lower()
    assert b.store.lookup_by_session_key(b.parent.session_key).session_id == "worker"
    if existing_human_goal:
        # A historical worker-origin foreground can now have genuine human
        # direction. No text inference or erasure of lineage authorizes routing.
        b.db.append_message("worker", "user", "Continue the current human goal")
        goals.save_goal("worker", goals.GoalState(goal="current human goal"))
    before = b.db.get_messages("worker")
    goal = b.db.get_meta("goal:worker")
    result = asyncio.run(b.runner._deliver_completion_notification("own process finished", _completion(b)))
    assert result is True
    assert b.resolved == ["worker"]
    assert b.store.lookup_by_session_key(b.parent.session_key).session_id == "worker"
    assert b.db.get_session("worker")["ended_at"] is None
    assert b.db.get_session("worker")["model_config"]
    assert b.db.get_messages("worker") == before
    assert b.db.get_meta("goal:worker") == goal
    assert goals.load_goal("worker").goal == ("current human goal" if existing_human_goal else "worker task")


def test_same_worker_on_another_route_is_not_authority_for_this_route(native_boundary):
    b = native_boundary
    other_source = SessionSource(platform=Platform.TELEGRAM, chat_id="999", user_id="999", chat_type="dm")
    other = b.store.get_or_create_session(other_source)
    b.store.switch_session(other.session_key, "worker")
    assert asyncio.run(b.runner._deliver_completion_notification("wrong route", _completion(b))) is None
    _assert_foreground_unchanged(b)
    assert b.events == []


def test_receiver_rechecks_after_bound_worker_preflight_and_new(native_boundary):
    b = native_boundary
    b.store.switch_session(b.parent.session_key, "worker")
    assert asyncio.run(b.runner._classify_completion_target("worker", b.parent.session_key)) == "deliver"
    fresh = b.store.reset_session(b.parent.session_key)
    fresh_id = fresh.session_id
    # Already-queued pin from BEFORE /new bypasses producer preflight.
    event = MessageEvent(text="late own process", source=b.source, internal=True,
                         metadata={"gateway_session_id": "worker"})
    asyncio.run(b.runner.adapters[Platform.TELEGRAM].handle_message(event))
    assert b.resolved == []
    assert b.store.lookup_by_session_key(b.parent.session_key).session_id == fresh_id
    assert b.db.get_session("worker")["end_reason"] == "session_reset"
    assert b.db.get_session(fresh_id)["ended_at"] is None


@pytest.mark.parametrize("advance_route", [False, True])
def test_resumed_foreground_delayed_completion_survives_rotation(native_boundary, monkeypatch, advance_route):
    b = native_boundary
    b.db.record_gateway_session_peer(
        "worker", source="telegram", user_id="123", chat_id="123",
        chat_type="dm", thread_id="456", session_key=b.parent.session_key,
    )
    for hook in ("_release_running_agent_state", "_clear_conversation_scope", "_evict_cached_agent"):
        monkeypatch.setattr(b.runner, hook, lambda *_a, **_kw: None)
    confirmation = asyncio.run(b.runner._handle_resume_command(
        MessageEvent(text="/resume worker", source=b.source)))
    assert "resumed" in confirmation.lower()
    provenance = b.db.get_session("worker")["model_config"]
    b.db.publish_compression_child(
        parent_session_id="worker", child_session_id="foreground-tip",
        source="telegram", messages=[{"role": "user", "content": "Continue the human goal"}],
        model_config={"max_iterations": 250}, require_compression_lease=False,
    )
    assert b.db.get_compression_tip("worker") == "foreground-tip"
    if advance_route:
        assert b.store.advance_compression_session(b.parent.session_key, "worker", "foreground-tip")
    result = asyncio.run(b.runner._deliver_completion_notification("delayed own process", _completion(b)))
    assert result is True
    assert b.store.lookup_by_session_key(b.parent.session_key).session_id == "foreground-tip"
    assert b.db.get_session("foreground-tip")["ended_at"] is None
    assert b.resolved == ["foreground-tip"]
    assert b.db.get_session("worker")["model_config"] == provenance


@pytest.mark.parametrize("route", ["worker", "middle", "foreground-tip"])
def test_resumed_owner_pin_survives_multiple_native_rotations(native_boundary, route):
    b = native_boundary
    b.store.switch_session(b.parent.session_key, "worker")
    for parent, child in (("worker", "middle"), ("middle", "foreground-tip")):
        assert b.db.try_acquire_compression_lock(parent, "compressor")
        b.db.publish_compression_child(
            parent_session_id=parent, child_session_id=child, source="telegram",
            model_config={"max_iterations": 250}, compression_lock_holder="compressor",
            messages=[{"role": "user", "content": "Continue the human goal"}],
        )
    if route != "worker":
        b.store.advance_compression_session(b.parent.session_key, "worker", route)
    goals.save_goal("foreground-tip", goals.GoalState(goal="current human goal"))
    assert asyncio.run(b.runner._deliver_completion_notification("backlog", _completion(b))) is True
    assert b.resolved == ["foreground-tip"]
    assert goals.load_goal("foreground-tip").goal == "current human goal"
    assert b.db.get_messages("foreground-tip")[0]["content"] == "Continue the human goal"


@pytest.mark.parametrize("owner", ["worker", "grandchild", "legacy-worker"])
@pytest.mark.parametrize("foreign_route", [False, True])
def test_worker_compression_is_not_foreground_adoption(native_boundary, owner, foreign_route):
    b = native_boundary
    if owner == "grandchild":
        b.db.create_session(owner, source="subagent", parent_session_id="worker",
                            model_config={"_delegate_from": "worker"})
    elif owner == "legacy-worker":
        b.db.create_session(owner, source="subagent", parent_session_id=b.parent_id)
    # Even a marker-free tip is insufficient: this event's actual route must
    # own the pin's compression lineage, not merely inherit transport metadata.
    assert b.db.try_acquire_compression_lock(owner, "compressor")
    b.db.publish_compression_child(
        parent_session_id=owner, child_session_id="worker-tip", source="telegram",
        model_config={"max_iterations": 250}, compression_lock_holder="compressor",
        messages=[{"role": "user", "content": "worker-only summary"}],
    )
    if foreign_route:
        other = b.store.get_or_create_session(SessionSource(
            platform=Platform.TELEGRAM, chat_id="999", user_id="999", chat_type="dm"))
        b.store.switch_session(other.session_key, "worker-tip")
    assert asyncio.run(b.runner._deliver_completion_notification("wrong route", _completion(b, owner))) is None
    event = MessageEvent(text="queued worker", source=b.source, internal=True,
                         metadata={"gateway_session_id": owner})
    asyncio.run(b.runner.adapters[Platform.TELEGRAM].handle_message(event))
    _assert_foreground_unchanged(b)
    assert b.resolved == []


@pytest.mark.parametrize("config", [{"_delegate_from": "parent"}, '{"_delegate_from":"parent"}'])
def test_native_marker_representation(config):
    assert GatewayRunner._is_delegated_completion_owner({"source": "gateway", "model_config": config})


@pytest.mark.parametrize("config", [None, "{}", "[]", "malformed", {"_reset_from": "parent"}])
def test_parentage_without_delegation_marker_is_not_worker(config):
    assert not GatewayRunner._is_delegated_completion_owner({
        "source": "telegram", "parent_session_id": "parent", "model_config": config,
    })
