"""Completion ownership for GatewayRunner: the session a pinned completion belongs to.

Split out of ``gateway/run_notifications.py`` (itself split out of ``gateway/run.py``); bound onto
``GatewayRunner`` through ``GatewayNotificationsMixin``. A completion must belong to the foreground session or its verified compression
continuation. Delegated provenance never grants authority to forward a worker
completion; explicit human adoption of a worker remains a valid foreground route.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional, Sequence, cast

from gateway.session import SessionEntry

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

class GatewayNotificationOwnershipMixin:
    """Ownership proof and route resolution for pinned async and process completions."""

    @staticmethod
    def _is_delegated_completion_owner(row: dict) -> bool:
        """Whether durable worker provenance makes a row ineligible to claim a route.

        Peer refresh can replace ``source`` with a transport value while the
        delegation marker survives worker compression.  A bare parent link is
        deliberately not a worker marker: normal compression also has one.
        """
        if row.get("source") == "subagent":
            return True
        config = row.get("model_config") or {}
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except (TypeError, ValueError):
                return False
        return isinstance(config, dict) and bool(config.get("_delegate_from"))

    async def _resolve_compression_lineage_target(
        self, session_db: Any, session_entry: Optional[SessionEntry], pinned_session_id: str,
    ) -> tuple[str, Optional[str]]:
        """Distinguish uncertain compression ownership from a proven foreign route."""
        try:
            target_session_id = await session_db.get_compression_tip(pinned_session_id)
            if not target_session_id or target_session_id == pinned_session_id:
                return "retry", None  # Rotation may not have published its continuation yet.
            tip_row = await session_db.get_session(target_session_id)
            if tip_row is not None and self._is_delegated_completion_owner(tip_row):
                return "terminal", None
            if tip_row is None or tip_row.get("ended_at"):
                return "retry", None
            if session_entry is None or session_entry.session_id in {pinned_session_id, target_session_id}:
                return "deliver", target_session_id
            # Across several rotations, accept a stale route only when its own tip is the same live target.
            route_row = await session_db.get_session(session_entry.session_id)
            if route_row is None:
                return "retry", None
            if route_row.get("ended_at") and route_row.get("end_reason") == "compression":
                route_tip = await session_db.get_compression_tip(session_entry.session_id)
                if not route_tip or route_tip == session_entry.session_id:
                    return "retry", None
                if route_tip == target_session_id:
                    return "deliver", target_session_id
        except Exception:
            logger.debug("Async-delegation compression ownership lookup failed for %s", pinned_session_id, exc_info=True)
            return "retry", None
        logger.warning(
            "Async-delegation completion for compression lineage %s -> %s "
            "does not own current route %s; dropping injection.",
            pinned_session_id, target_session_id, session_entry.session_id,
        )
        return "terminal", None

    async def _lookup_completion_owner(self, session_id: str, current_session_id: str = ""):
        """Verify this execution owns the foreground; never forward a worker completion."""
        from gateway.run import _USER_BOUNDARY_END_REASONS
        session_db = cast(Any, self)._session_db
        if session_db is None:
            return "retry", session_id, None
        try:
            row = await session_db.get_session(session_id)
        except Exception:
            logger.debug("Completion ownership lookup failed for %s", session_id, exc_info=True)
            return "retry", session_id, None
        if row is None:
            return "terminal", session_id, None
        if row.get("ended_at") and row.get("end_reason") in _USER_BOUNDARY_END_REASONS:
            return "terminal", session_id, row
        if session_id == current_session_id:
            return "deliver", session_id, row
        config = row.get("model_config")
        try:
            if isinstance(config, str):
                config = json.loads(config)
        except (TypeError, ValueError):
            return "terminal", session_id, None
        if config is None:
            config = {}
        if not isinstance(config, dict):
            return "terminal", session_id, None
        if "_delegate_from" in config and (
            not isinstance(config["_delegate_from"], str) or not config["_delegate_from"].strip()
        ):
            return "terminal", session_id, None
        if self._is_delegated_completion_owner(row):
            if row.get("ended_at") and row.get("end_reason") == "compression":
                try:
                    tip_id = await session_db.get_compression_tip(session_id)
                    tip = await session_db.get_session(tip_id) if tip_id and tip_id != session_id else None
                except Exception:
                    return "retry", session_id, None
                if tip is not None and not self._is_delegated_completion_owner(tip):
                    return "deliver", session_id, row
            return "terminal", session_id, row
        return "deliver", session_id, row

    async def _resolve_async_delegation_session(
        self, session_entry: SessionEntry, pinned_session_id: str, proven_session_id: str = "",
        completion_ids: Sequence[str] = (),
    ) -> Optional[SessionEntry]:
        """Resolve an async completion to its verified owning gateway session.

        Follow compression-rotation lineage (parent row ended, child continues), but never let a
        late completion override an unrelated /new or restored route. Unknown ownership fails
        closed; the result stays in the delegation records. An owner that cannot be confirmed right
        now (``retry``) sends the completions admitted with the event back to the delivery queue.

        Worker provenance never forwards a completion to another foreground.
        Explicit human adoption and its verified compression continuations are
        valid routes; malformed provenance remains fail-closed.
        """
        from gateway.run import _USER_BOUNDARY_END_REASONS
        session_db = cast(Any, self._session_db)
        if session_db is None:
            logger.warning(
                "Async-delegation completion has no session database; "
                "dropping injection (#55578 fail-closed)."
            )
            return None
        if proven_session_id and proven_session_id == session_entry.session_id:
            # Admission already settled the durable claim on this proof; another read could only lose it.
            return session_entry
        # Snapshot before ownership awaits: a concurrent /stop or /new wins.
        run_generation = self._current_session_run_generation(session_entry.session_key)
        verdict, pinned_session_id, pinned_row = await self._lookup_completion_owner(
            pinned_session_id, session_entry.session_id,
        )
        if verdict != "deliver" or pinned_row is None:
            return await self._requeue_uncertain_completion(verdict, completion_ids)
        target_session_id = pinned_session_id
        follows_compression = False
        if pinned_row.get("ended_at"):
            _end_reason = str(pinned_row.get("end_reason") or "")
            if _end_reason in _USER_BOUNDARY_END_REASONS:
                logger.warning(
                    "Async-delegation completion pinned to user-closed session %s "
                    "(end_reason=%r); dropping injection instead of resurrecting it "
                    "(#55578 fail-closed).", pinned_session_id, _end_reason,
                )
                return None
            if _end_reason != "compression":
                # Idle/timeout end (scale-to-zero norm): the chat route is still valid, so deliver to its
                # current session rather than drop (the row would be acked then silently lost).
                logger.info(
                    "Async-delegation completion pinned to %s-ended session %s; "
                    "retargeting to the chat's current session %s.",
                    _end_reason or "idle", pinned_session_id, session_entry.session_id,
                )
                return session_entry
            follows_compression = True
            verdict, target_session_id = await self._resolve_compression_lineage_target(
                session_db, session_entry, pinned_session_id,
            )
            if verdict != "deliver" or target_session_id is None:
                return await self._requeue_uncertain_completion(verdict, completion_ids)
        if target_session_id == session_entry.session_id:
            return session_entry
        prior_session_id = session_entry.session_id
        if not self._is_session_run_current(session_entry.session_key, run_generation):
            logger.warning(
                "Async-delegation completion for routing key %s was invalidated while resolving pinned "
                "session %s; leaving the route on %s and dropping injection.",
                session_entry.session_key, pinned_session_id, prior_session_id,
            )
            return None
        if follows_compression:
            switched = await self.async_session_store.advance_compression_session(
                session_entry.session_key, prior_session_id, target_session_id,
            )
        else:
            # CAS on the session this completion resolved against: a route replaced meanwhile
            # (/new, /resume) wins over the stale completion.
            switched = await self.async_session_store.switch_session(
                session_entry.session_key, target_session_id, expected_session_id=prior_session_id,
            )
        if switched is None:
            logger.warning(
                "Async-delegation completion could not bind routing key %s to "
                "owning session %s (route moved or unknown); dropping injection.",
                session_entry.session_key, target_session_id,
            )
            return None
        logger.info(
            "Pinned async-delegation completion to owning session %s (was %s) for routing key %s (#57498)",
            target_session_id, prior_session_id, session_entry.session_key,
        )
        return switched

    async def _requeue_uncertain_completion(self, verdict: str, completion_ids: Sequence[str]) -> None:
        """Offer admitted completions again when the turn could not confirm their owner (``retry``),
        so the next delivery re-runs the pre-flight; boundary verdicts stay dropped."""
        if verdict != "retry" or not completion_ids:
            return
        from tools.async_delegation import requeue_admitted_completion
        from tools.process_registry import process_registry
        runner = cast(Any, self)
        with runner._completion_delivery_lock:
            for delegation_id in completion_ids:
                runner._completion_deliveries_delivered.pop(("async_delegation", delegation_id, ""), None)
        for delegation_id in completion_ids:
            try:
                requeued = await asyncio.to_thread(
                    requeue_admitted_completion, delegation_id, process_registry.completion_queue)
            except Exception:
                logger.debug("Could not requeue admitted completion %s", delegation_id, exc_info=True)
                requeued = False
            logger.warning(
                "Async-delegation completion %s: owner not confirmed after admission; %s.", delegation_id,
                "offered again" if requeued else "not offered again (result remains queryable)",
            )
