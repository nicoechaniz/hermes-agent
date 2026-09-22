"""Kimi Code and Z.AI endpoint auto-detection, LM Studio base-URL normalization.

Re-exported from ``hermes_cli/auth.py`` (patch targets unchanged); origin helpers are imported
lazily per function so ``hermes_cli.auth.<helper>`` patches still intercept and no cycle forms.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_cli.auth_constants import AuthError, httpx
from utils import atomic_json_write

logger = logging.getLogger("hermes_cli.auth")

# In-process negative cache for Z.AI endpoint detection, keyed by key hash: a failed probe is not
# retried for this long (a success persists to auth.json instead).
_ZAI_PROBE_FAILURE_TTL_SECONDS = 300
_zai_probe_failed_until: Dict[str, float] = {}

# "sk-kimi-" keys only work on api.kimi.com/coding; legacy moonshot keys use the old default.
# NO /v1 suffix: the anthropic SDK appends "/v1/messages" itself ("/coding/v1" would 404).
KIMI_CODE_BASE_URL = "https://api.kimi.com/coding"

# Kimi Code owns these values.  Hermes only consumes the credentials produced by
# ``kimi login``; it never initiates this OAuth flow itself.
KIMI_CODE_CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
KIMI_CODE_OAUTH_HOST = "https://auth.kimi.com"
KIMI_CODE_CLI_USER_AGENT = "kimi-code-cli"


def _kimi_code_home() -> Path:
    """Current Kimi Code home, honoring its documented override."""
    configured = os.getenv("KIMI_CODE_HOME", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".kimi-code"


def _kimi_cli_store_home() -> Path:
    """Use one coherent current/legacy store, preferring the current CLI."""
    current_home = _kimi_code_home()
    if os.getenv("KIMI_CODE_HOME", "").strip():
        return current_home
    legacy_home = Path.home() / ".kimi"
    current_credentials = current_home / "credentials" / "kimi-code.json"
    legacy_credentials = legacy_home / "credentials" / "kimi-code.json"
    return current_home if current_credentials.exists() or not legacy_credentials.exists() else legacy_home


def _kimi_cli_credentials_path() -> Path:
    return _kimi_cli_store_home() / "credentials" / "kimi-code.json"


def _kimi_cli_device_id_path() -> Path:
    return _kimi_cli_store_home() / "device_id"


def _kimi_cli_version() -> str:
    """Best-effort installed CLI version; do not make auth depend on its binary."""
    try:
        kimi_bin = shutil.which("kimi")
        if kimi_bin:
            result = subprocess.run(
                [kimi_bin, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            for part in result.stdout.strip().split():
                part = part.strip().rstrip(",")
                if part and part[0].isdigit():
                    return part
    except Exception:
        logger.debug("Could not discover Kimi CLI version", exc_info=True)
    return "unknown"


def _read_kimi_cli_credentials() -> Dict[str, Any]:
    """Read the official CLI's JSON store without exposing its secret values."""
    credential_path = _kimi_cli_credentials_path()
    if not credential_path.exists():
        raise AuthError("Kimi CLI credentials not found. Run `kimi login` first.", provider="kimi-coding",
                        code="kimi_auth_missing")
    try:
        payload = json.loads(credential_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuthError("Kimi CLI credentials could not be read. Run `kimi login` again.", provider="kimi-coding",
                        code="kimi_auth_read_failed", relogin_required=True) from exc
    if not isinstance(payload, dict):
        raise AuthError("Kimi CLI credentials are invalid. Run `kimi login` again.", provider="kimi-coding",
                        code="kimi_auth_invalid", relogin_required=True)
    return payload


def _save_kimi_cli_credentials(tokens: Dict[str, Any]) -> Path:
    """Atomically persist a rotated OAuth chain with owner-only permissions."""
    credential_path = _kimi_cli_credentials_path()
    credential_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(credential_path, tokens, indent=2, mode=0o600, fsync_dir=True)
    return credential_path


def _kimi_oauth_token_is_expired(expires_at: Any, skew_seconds: int = 300) -> bool:
    try:
        expiry = float(expires_at)
    except (TypeError, ValueError):
        return True
    return expiry <= time.time() + max(0, skew_seconds)


def _kimi_cli_result(token: str, base_url: str, source: str) -> Dict[str, Any]:
    return {
        "provider": "kimi-coding", "api_key": token, "base_url": base_url.rstrip("/"),
        "source": source, "auth_file": str(_kimi_cli_credentials_path()), "kimi_cli_oauth": True,
    }


def _refresh_kimi_cli_credentials(tokens: Dict[str, Any], *, base_url: str) -> Dict[str, Any]:
    refresh_token = str(tokens.get("refresh_token") or "").strip()
    if not refresh_token:
        raise AuthError("Kimi CLI OAuth credentials need `kimi login` again.", provider="kimi-coding",
                        code="kimi_oauth_missing_refresh_token", relogin_required=True)
    try:
        with httpx.Client(timeout=httpx.Timeout(20.0), headers={"Accept": "application/json"}) as client:
            response = client.post(
                f"{KIMI_CODE_OAUTH_HOST}/api/oauth/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": KIMI_CODE_CLIENT_ID},
            )
    except Exception as exc:
        raise AuthError("Kimi token refresh failed.", provider="kimi-coding", code="kimi_oauth_refresh_failed") from exc
    if response.status_code != 200:
        relogin = response.status_code in {401, 403}
        raise AuthError(f"Kimi token refresh failed with status {response.status_code}.", provider="kimi-coding",
                        code="kimi_oauth_refresh_failed", relogin_required=relogin)
    try:
        refreshed = response.json()
    except Exception as exc:
        raise AuthError("Kimi token refresh returned invalid JSON.", provider="kimi-coding",
                        code="kimi_oauth_refresh_invalid_json", relogin_required=True) from exc
    access_token = refreshed.get("access_token") if isinstance(refreshed, dict) else None
    if not isinstance(access_token, str) or not access_token.strip():
        raise AuthError("Kimi token refresh did not return an access token.", provider="kimi-coding",
                        code="kimi_oauth_refresh_invalid_payload", relogin_required=True)
    updated = dict(tokens)
    updated["access_token"] = access_token.strip()
    updated["refresh_token"] = str(refreshed.get("refresh_token") or refresh_token).strip()
    try:
        expires_in = float(refreshed.get("expires_in"))
    except (TypeError, ValueError):
        expires_in = 3600.0
    updated["expires_at"] = time.time() + max(1.0, expires_in)
    for key in ("expires_in", "scope", "token_type"):
        if key in refreshed:
            updated[key] = refreshed[key]
    _save_kimi_cli_credentials(updated)
    return _kimi_cli_result(updated["access_token"], base_url, "kimi-cli-oauth-refresh")


def resolve_kimi_cli_oauth_credentials(*, base_url: str, force_refresh: bool = False) -> Dict[str, Any]:
    """Resolve and proactively refresh credentials owned by the official Kimi CLI."""
    tokens = _read_kimi_cli_credentials()
    access_token = str(tokens.get("access_token") or "").strip()
    if access_token and not force_refresh and not _kimi_oauth_token_is_expired(tokens.get("expires_at")):
        return _kimi_cli_result(access_token, base_url, "kimi-cli-oauth")
    return _refresh_kimi_cli_credentials(tokens, base_url=base_url)


def get_kimi_cli_oauth_status() -> Dict[str, Any]:
    """Read-only provider status for picker discovery; this never refreshes or writes."""
    try:
        tokens = _read_kimi_cli_credentials()
    except AuthError:
        return {"configured": False, "logged_in": False, "provider": "kimi-coding"}
    access_token = str(tokens.get("access_token") or "").strip()
    refresh_token = str(tokens.get("refresh_token") or "").strip()
    usable = bool(access_token or refresh_token)
    return {
        "configured": usable, "logged_in": usable, "provider": "kimi-coding",
        "key_source": "kimi-cli-oauth" if usable else "", "base_url": KIMI_CODE_BASE_URL,
        "token_expired": _kimi_oauth_token_is_expired(tokens.get("expires_at"), skew_seconds=0),
    }


def kimi_cli_model_ids() -> list[str]:
    """Read the official CLI's local Coding-plan aliases without a network request."""
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Hermes requires Python 3.11+
        return []
    homes = [_kimi_cli_store_home()]
    if not os.getenv("KIMI_CODE_HOME", "").strip():
        homes.extend([_kimi_code_home(), Path.home() / ".kimi"])
    for home in dict.fromkeys(homes):
        try:
            config = tomllib.loads((home / "config.toml").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        models = config.get("models")
        if not isinstance(models, dict):
            continue
        default = config.get("default_model")
        keys = ([default] if isinstance(default, str) else []) + [key for key in models if key != default]
        result, seen = [], set()
        for key in keys:
            entry = models.get(key)
            if not isinstance(entry, dict) or entry.get("provider") != "managed:kimi-code":
                continue
            model_id = str(entry.get("model") or "").strip()
            if model_id and model_id.lower() not in seen:
                seen.add(model_id.lower())
                result.append(model_id)
        if result:
            return result
    return []


def kimi_coding_default_headers() -> Dict[str, str]:
    """Official CLI identity headers for a token read from its own credential store."""
    device_id = ""
    try:
        device_id = _kimi_cli_device_id_path().read_text(encoding="utf-8").strip()
    except OSError:
        pass
    version = _kimi_cli_version()
    headers = {
        "User-Agent": f"{KIMI_CODE_CLI_USER_AGENT}/{version}", "X-Msh-Platform": "kimi_cli",
        "X-Msh-Version": version, "X-Msh-Device-Name": platform.node(),
        "X-Msh-Device-Model": platform.machine(), "X-Msh-Os-Version": platform.version(),
    }
    if device_id:
        headers["X-Msh-Device-Id"] = device_id
    return headers


def _resolve_kimi_base_url(api_key: str, default_url: str, env_override: str) -> str:
    """Kimi base URL from the key prefix; an explicit KIMI_BASE_URL always wins."""
    if env_override:
        return env_override
    if api_key and api_key.startswith("sk-kimi-"):
        return KIMI_CODE_BASE_URL
    return default_url


# Z.AI bills general/coding plans and global/China endpoints separately ("Insufficient balance" on
# the wrong one), so probe once and cache. Candidate models are tried in order: newer coding-plan
# accounts may only have recent GLM slugs, older ones still glm-4.7.
_ZAI_CODING_PROBE_MODELS = ["glm-5.3", "glm-5.3-flash", "glm-5.2", "glm-5.1", "glm-5v-turbo", "glm-4.7"]
ZAI_ENDPOINTS = [
    # (id, base_url, probe_models, label)
    ("global",        "https://api.z.ai/api/paas/v4",        ["glm-5"],   "Global"),
    ("cn",            "https://open.bigmodel.cn/api/paas/v4", ["glm-5"],   "China"),
    ("coding-global", "https://api.z.ai/api/coding/paas/v4",  _ZAI_CODING_PROBE_MODELS, "Global (Coding Plan)"),
    ("coding-cn",     "https://open.bigmodel.cn/api/coding/paas/v4", _ZAI_CODING_PROBE_MODELS, "China (Coding Plan)"),
]


def _probe_single_zai_endpoint(api_key: str, endpoint: tuple, timeout: float) -> Optional[Dict[str, str]]:
    """Probe one Z.AI endpoint, trying its candidate models in order; None when none succeeds."""
    ep_id, base_url, probe_models, label = endpoint
    for model in probe_models:
        try:
            resp = httpx.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "stream": False, "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]},
                timeout=timeout,
            )
            if resp.status_code == 200:
                logger.debug("Z.AI endpoint probe: %s (%s) model=%s OK", ep_id, base_url, model)
                return {"id": ep_id, "base_url": base_url, "model": model, "label": label}
            logger.debug("Z.AI endpoint probe: %s model=%s returned %s", ep_id, model, resp.status_code)
        except Exception as exc:
            logger.debug("Z.AI endpoint probe: %s model=%s failed: %s", ep_id, model, exc)
    return None


def detect_zai_endpoint(api_key: str, timeout: float = 8.0) -> Optional[Dict[str, str]]:
    """Probe z.ai endpoints in parallel; first working one in ZAI_ENDPOINTS priority order, or None."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    # No `with`: it would join ALL probes on exit, defeating the early return below.
    pool = ThreadPoolExecutor(max_workers=len(ZAI_ENDPOINTS))
    try:
        futures = {pool.submit(_probe_single_zai_endpoint, api_key, ep, timeout): ep[0] for ep in ZAI_ENDPOINTS}
        by_id = {ep_id: f for f, ep_id in futures.items()}
        results: Dict[str, Dict[str, str]] = {}

        def _first_ready(require_done: bool) -> Optional[Dict[str, str]]:
            # Walk endpoints in PRIORITY order; a lower-priority success only wins once every
            # higher-priority probe has finished without success.
            for ep in ZAI_ENDPOINTS:
                if require_done and not by_id[ep[0]].done():
                    return None  # a higher-priority probe is still in flight
                if ep[0] in results:
                    return results[ep[0]]
            return None

        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    results[futures[future]] = result
            except Exception:
                pass
            winner = _first_ready(require_done=True)
            if winner is not None:
                return winner
        return _first_ready(require_done=False)
    finally:
        pool.shutdown(wait=False)


def _resolve_zai_base_url(api_key: str, default_url: str, env_override: str) -> str:
    """Z.AI base URL by probing endpoints; an explicit GLM_BASE_URL always wins.

    The detected endpoint is cached in provider state (auth.json) keyed on a hash of the API key so
    subsequent starts skip the probe.
    """
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _load_provider_state, _save_auth_store, _store_provider_state, detect_zai_endpoint
    if env_override:
        return env_override
    # No key -> don't probe (N×M 401s); auxiliary-client auto-detection hits this for everyone.
    if not api_key:
        return default_url

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    state = _load_provider_state(_load_auth_store(), "zai") or {}
    cached = state.get("detected_endpoint")
    if isinstance(cached, dict) and cached.get("base_url") and cached.get("key_hash", "") == key_hash:
        logger.debug("Z.AI: using cached endpoint %s", cached["base_url"])
        return cached["base_url"]
    # Only a success is persisted, so a failing key (429/401 on every endpoint) would re-run the
    # four chat-completion probes on every credential-pool load — dozens of times per picker open.
    if _zai_probe_failed_until.get(key_hash, 0.0) > time.time():
        return default_url

    # Probe — may take up to ~8s per endpoint.
    detected = detect_zai_endpoint(api_key)
    if not (detected and detected.get("base_url")):
        logger.debug("Z.AI: probe failed, falling back to default %s", default_url)
        _zai_probe_failed_until[key_hash] = time.time() + _ZAI_PROBE_FAILURE_TTL_SECONDS
        return default_url

    detected_endpoint = {
        "base_url": detected["base_url"], "endpoint_id": detected.get("id", ""),
        "model": detected.get("model", ""), "label": detected.get("label", ""),
        "key_hash": key_hash,
    }
    # Persist failure must not break resolution; worst case the next start re-probes.
    try:
        with _auth_store_lock():
            auth_store = _load_auth_store()  # reload under lock to avoid overwriting concurrent changes
            state_under_lock = _load_provider_state(auth_store, "zai") or {}
            state_under_lock["detected_endpoint"] = detected_endpoint
            # set_active=False: runs from credential-pool env seeding; must not flip active provider.
            _store_provider_state(auth_store, "zai", state_under_lock, set_active=False)
            _save_auth_store(auth_store)
    except Exception as exc:
        logger.warning("Z.AI: could not persist detected endpoint (%s); will re-probe next start", exc)
    logger.info("Z.AI: auto-detected endpoint %s (%s)", detected["label"], detected["base_url"])
    return detected["base_url"]


def _normalize_lmstudio_runtime_base_url(base_url: str) -> str:
    """Return the OpenAI-compatible LM Studio runtime base URL.

    LM Studio's native management API lives under ``/api/v1`` while its OpenAI-compatible chat
    endpoint lives under ``/v1``; users paste either form, so normalize before the SDK appends
    ``/chat/completions``.
    """
    root = str(base_url or "").strip().rstrip("/")
    for suffix in ("/api/v1", "/api", "/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)].rstrip("/")
            break
    return (root or "http://127.0.0.1:1234") + "/v1"
