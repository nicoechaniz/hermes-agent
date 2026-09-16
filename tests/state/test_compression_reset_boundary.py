"""A /new child is a separate conversation, even below a stale compressed row."""
import pytest

from hermes_state import SessionDB


@pytest.fixture
def reset_chain(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent", source="telegram", session_key="old-route")
    db.append_message("parent", "user", "old goal")
    db.end_session("parent", "compression")
    db.create_session("reset", source="telegram", parent_session_id="parent",
                      model_config={"_reset_from": "parent"})
    db.append_message("reset", "user", "new goal")
    yield db
    db.close()


def test_reset_is_not_live_compression_child(reset_chain):
    assert reset_chain.find_live_compression_child("parent") is None


@pytest.mark.parametrize("start", ["parent", "reset"])
def test_reset_is_not_in_compression_lineage(reset_chain, start):
    assert reset_chain.get_compression_lineage(start) == [start]


def test_reset_has_independent_turn_lease(reset_chain):
    db = reset_chain
    assert db.try_acquire_session_turn_lease("parent", "old-turn")
    assert db.try_acquire_session_turn_lease("reset", "new-turn")


def test_reset_can_start_its_own_compression_lineage(reset_chain):
    db = reset_chain
    assert db.try_acquire_compression_lock("reset", "compressor")
    db.publish_compression_child(
        parent_session_id="reset", child_session_id="new-tip", source="telegram",
        model_config={"max_iterations": 250},
        messages=[{"role": "user", "content": "new goal summary"}],
        compression_lock_holder="compressor",
    )
    assert db.get_compression_tip("parent") == "parent"
    assert db.get_compression_tip("reset") == "new-tip"
    assert db.get_compression_lineage("reset") == ["reset", "new-tip"]
    assert db.get_compression_lineage("new-tip") == ["reset", "new-tip"]
    assert db._session_turn_lease_key("new-tip") == "reset"


def test_reset_does_not_inherit_compressed_parents_peer(reset_chain):
    assert reset_chain.get_session("reset")["session_key"] is None


def test_reset_peer_refresh_does_not_rewrite_compression_parent(reset_chain):
    db = reset_chain
    db.record_gateway_session_peer("reset", source="telegram", session_key="new-route",
                                   include_compression_ancestors=True)
    assert db.get_session("parent")["session_key"] == "old-route"
    assert db.get_session("reset")["session_key"] == "new-route"


def test_reset_cannot_take_compression_ancestors_title(reset_chain):
    db = reset_chain
    db.set_session_title("parent", "Old conversation")
    with pytest.raises(ValueError, match="already in use"):
        db.set_session_title("reset", "Old conversation")
    assert db.get_session("parent")["title"] == "Old conversation"


@pytest.mark.parametrize("field", ["archived", "pinned", "hidden", "read"])
@pytest.mark.parametrize("start,other", [("parent", "reset"), ("reset", "parent")])
def test_compression_scoped_flags_do_not_cross_reset(reset_chain, field, start, other):
    db = reset_chain
    column = "last_read_at" if field == "read" else field
    before = db.get_session(other)[column]
    assert getattr(db, "set_session_" + field)(start, True)
    assert db.get_session(other)[column] == before
    assert db.get_session(start)[column]


def test_reset_is_not_reaped_as_orphaned_compression(reset_chain):
    db = reset_chain
    # Old enough for the bounded native orphan sweep, with no model calls.
    db._conn.execute("UPDATE sessions SET started_at = 1 WHERE id = 'reset'")
    db._conn.commit()
    assert db.finalize_orphaned_compression_sessions() == 0
    assert db.get_session("reset")["ended_at"] is None


def test_reset_activity_does_not_rank_old_compression_conversation(reset_chain):
    db = reset_chain
    db.create_session("between", source="telegram")
    db._conn.execute("UPDATE sessions SET started_at = 1, last_activity_at = 1")
    db._conn.execute("UPDATE sessions SET started_at = 50, last_activity_at = 50 WHERE id = 'between'")
    db._conn.execute("UPDATE messages SET timestamp = 1 WHERE session_id = 'parent'")
    db._conn.execute("UPDATE messages SET timestamp = 100 WHERE session_id = 'reset'")
    db._conn.commit()
    rows = db.list_sessions_rich(order_by_last_active=True, project_compression_tips=False)
    assert [row["id"] for row in rows] == ["reset", "between", "parent"]
