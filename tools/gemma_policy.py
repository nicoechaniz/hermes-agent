#!/usr/bin/env python3
"""gemma_policy.py — 5-layer Hermes mitigation policy for Gemma-Andy upstream.

Pure policy filter: no HTTP calls, no Ollama, no embodied-service dependencies.
Consumes a raw user intent string and returns either:
  - policy_handled_upstream  (L2 scope cut or L3 ambiguity cut)
  - embodied_ready           (L5→L1→L4 pipeline result with sub-intents)

Layers (canonical order):
  L2 — scope filter   : non-body intents → handled upstream
  L3 — ambiguity      : vague intents → handled upstream
  L5 — decompose      : multi-step intents → atomic sub-intents
  L1 — normalize      : per sub-intent, ES→EN imperative, canonical names
  L4 — narrow tools   : per sub-intent, classify category, narrow allowed_tools

Flow: L2 → L3 → L5 → (for each sub) → L1 → L4

Reference: Mar-IA-no/deamoncraft-gemma4-andy  mitigation/hermes_policy.py
"""
from __future__ import annotations

import os
import re
from typing import Optional


class GemmaPolicy:
    # ── L2: Out-of-scope regex ────────────────────────────────
    OUT_OF_SCOPE_REGEX = re.compile(
        r"\b(chiste|joke|cantam|hola|chau|buenas|gracias|de nada|"
        r"qué pensás|que pensas|qué te parece|que te parece|opinión|opinion|"
        r"explicame|explícame|definí|defini|definition|tell me about|"
        r"por qué|por que|why does|how does|qué es|que es|"
        r"sumar|restar|multiplicar|dividir|cuánto es|cuanto es|2\s*\+\s*2)\b",
        re.IGNORECASE,
    )

    # ── L3: Ambiguity tokens ──────────────────────────────────
    AMBIGUITY_TOKENS = re.compile(
        r"\b(algo entretenido|algo bueno|algo divertido|cualquier cosa|"
        r"something good|something fun|whatever|por ahí|por ahi|"
        r"hacé algo|hace algo|do something|haz algo|alrededor sin más|"
        r"andá por|anda por)\b",
        re.IGNORECASE,
    )

    # ── L4: Category keywords (most specific first) ───────────
    CATEGORY_KEYWORDS = [
        ("navigation",      ["andá", "anda ", "vení", "veni ", "venite", "follow", "go to", "goto", "alejate", "alejame", "flee", "seguime", "acercate", "come to", "come here", "ven aca", "ven acá", "find and approach", "approach", "find the player", "stop within", "stay within", "move away from", "move away", "move_away", "flee from", "flee_from", "get away from"]),
        ("equip",           ["equipá", "equipa", "equip", "ponete", "pongate"]),
        ("toss",            ["tirá", "tira", "toss", "drop", "dejá caer", "deja caer"]),
        ("pickup",          ["recogé", "recoge", "pickup", "agarrá", "agarra", "levantá", "levanta", "pick up"]),
        ("food",            ["comé", "comer", "comelo", "eat ", "drink", "bebé", "bebe", "morder", "ingerir"]),
        ("memory",          ["acordate", "marcá", "marca ", "recordá", "remember", "volvé a", "return to", "olvidá", "forget"]),
        ("mining",          ["minar", "mine ", "conseguí", "consegui", "gather", "dig "]),
        ("build",           ["construí", "construye", "construí ", "pongá", "place ", "build ", "make a "]),
        ("combat",          ["atacá", "ataca", "attack", "defendé", "defend", "raise_shield"]),
        ("inventory_query", ["inventario", "inventory", "decime qué tenés", "decime que tenes", "mostrame el inventario", "what do you have", "show inventory"]),
    ]

    CATEGORY_TOOLS = {
        "navigation":      ["scan_nearby", "goto", "follow", "stop_movement", "move_away"],
        "mining":          ["scan_nearby", "goto", "mine_block", "mine_blocks", "collect_drops", "get_inventory"],
        "equip":           ["get_inventory", "equip_item"],
        "toss":            ["get_inventory", "toss_item"],
        "pickup":          ["scan_nearby", "pickup_item", "get_inventory"],
        "inventory_query": ["get_inventory"],
        "memory":          ["remember_here", "goto_remembered_place", "forget_place", "get_inventory"],
        "food":            ["consume_food", "get_inventory"],
        "build":           ["scan_nearby", "goto", "place_block", "equip_item", "get_inventory"],
        "combat":          ["scan_nearby", "attack_entity", "flee_from", "raise_shield", "consume_food"],
    }
    COMMON_SAFE = ["ask_clarification", "report_execution_error"]
    GUARDIAN_AWARE_CATEGORIES = {"navigation", "combat", "default"}

    # ── L5: Decomposition ─────────────────────────────────────
    DECOMPOSE_CONNECTORS = re.compile(
        r"(\s+después\s+|\s+despues\s+|\s+luego\s+|\s+y después\s+|"
        r"\s+y luego\s+|\s+y después de\s+|\bthen\b|"
        r"(?<=[a-záéíóú])\.\s+(?=[A-ZÁÉÍÓÚ])|^\s*\d+[\.\)])",
        re.IGNORECASE,
    )
    CONSTRAINT_LEAD = re.compile(
        r"^(stop within|stay within|stay near|stay close|stay\s|"
        r"avoid|do not|don't|never|while|during|"
        r"without|keep|be careful|carefully|"
        r"sin (?:hacer|tocar|salir|hurt)|mantente|manténte|"
        r"evitando|cuidando|cuidado con)\b",
        re.IGNORECASE,
    )

    # ── L1: Verb map ES → EN imperative ───────────────────────
    VERB_MAP = [
        ("acordate de", "Remember"),
        ("acordate", "Remember"),
        ("recordá", "Remember"),
        ("marcá", "Mark"),
        ("marca ", "Mark "),
        ("volvé a", "Return to"),
        ("volvé", "Return"),
        ("vuelve a", "Return to"),
        ("alejate de", "Move away from"),
        ("alejate", "Move away from"),
        ("alejame", "Move away from"),
        ("andá a", "Go to"),
        ("andá", "Go to"),
        ("anda a", "Go to"),
        ("caminá", "Walk"),
        ("caminar", "Walk"),
        ("camina ", "Walk "),
        ("vení a", "Come to"),
        ("vení", "Come to"),
        ("venite a", "Come to"),
        ("venite", "Come to"),
        ("veni a", "Come to"),
        ("acercate", "Approach"),
        ("comé", "Eat"),
        ("comer ", "Eat "),
        ("minar", "Mine"),
        ("minas", "Mine"),
        ("conseguí", "Get"),
        ("consegui", "Get"),
        ("tirá", "Toss"),
        ("tira ", "Toss "),
        ("equipá", "Equip"),
        ("equipa", "Equip"),
        ("recogé", "Pick up"),
        ("recoge", "Pick up"),
        ("agarrá", "Pick up"),
        ("agarra ", "Pick up "),
        ("construí", "Build"),
        ("construye", "Build"),
        ("pongá", "Place"),
        ("atacá", "Attack"),
        ("ataca ", "Attack "),
        ("defendé", "Defend"),
        ("seguime", "Follow"),
        ("decime qué tenés", "Tell me what you have"),
        ("decime que tenes", "Tell me what you have"),
        ("mostrame el inventario", "Show your inventory"),
        ("hacé", "Do"),
        ("hace ", "Do "),
    ]

    def __init__(self, player_name: str | None = None, bot_name: str | None = None):
        self.player_name = player_name or os.getenv("HERMES_PLAYER_NAME", "player")
        self.bot_name = bot_name or os.getenv("HERMES_BOT_NAME", "minecraft_bot")

    # ─── Layer 2 ──────────────────────────────────────────────
    def is_out_of_scope(self, intent: str) -> tuple[bool, str | None]:
        if not intent:
            return False, None
        m = self.OUT_OF_SCOPE_REGEX.search(intent)
        return (bool(m), m.group(0) if m else None)

    # ─── Layer 3 ──────────────────────────────────────────────
    def is_ambiguous(self, intent: str) -> tuple[bool, str | None]:
        if not intent:
            return False, None
        m = self.AMBIGUITY_TOKENS.search(intent)
        return (bool(m), m.group(0) if m else None)

    # ─── Layer 4 ──────────────────────────────────────────────
    def classify_category(self, intent: str) -> str:
        low = (intent or "").lower()
        for cat, kws in self.CATEGORY_KEYWORDS:
            for kw in kws:
                if re.search(rf"\b{re.escape(kw)}", low):
                    return cat
        return "default"

    def get_allowed_tools(self, category: str) -> list[str] | None:
        if category == "default":
            return None
        base = list(self.CATEGORY_TOOLS[category]) + list(self.COMMON_SAFE)
        if category in self.GUARDIAN_AWARE_CATEGORIES:
            base.append("raise_guardian_event")
        return base

    # ─── Layer 5 ──────────────────────────────────────────────
    def decompose(self, intent: str) -> list[str]:
        """Split intent into atomic sub-intents.

        Conservative: only split when there are sequential PHYSICAL ACTIONS
        connected by temporal markers and the following clause starts with
        a new action verb (not a constraint/modifier).

        Constraint sub-intents (starting with "stop within", "avoid", etc.)
        are merged back into the previous sub-intent.
        """
        if not intent:
            return []
        if not self.DECOMPOSE_CONNECTORS.search(intent):
            return self._try_split_and(intent.strip())
        parts = self.DECOMPOSE_CONNECTORS.split(intent)
        CONNECTOR_TOKENS = {"después", "despues", "luego", "then", "y después", "y luego", "y", "y después de"}
        atomic_raw = []
        for p in parts:
            if not p:
                continue
            stripped = p.strip(" ,.;").strip()
            if not stripped:
                continue
            low = stripped.lower()
            if low in CONNECTOR_TOKENS:
                continue
            if re.match(r"^\d+[\.\)]?$", stripped):
                continue
            atomic_raw.append(stripped)
        if len(atomic_raw) <= 1:
            return self._try_split_and(intent.strip())
        # Constraint detection: merge constraint sub-intents back into the previous one
        merged: list[str] = [atomic_raw[0]]
        for sub in atomic_raw[1:]:
            if self.CONSTRAINT_LEAD.match(sub):
                merged[-1] = merged[-1].rstrip(" ,.;") + "; " + sub
            else:
                merged.append(sub)
        if len(merged) > 1:
            return merged
        return self._try_split_and(intent.strip())

    def _try_split_and(self, intent: str) -> list[str]:
        """Fallback split on ' and ' / ' y ' when both sides look like complete clauses."""
        for connector in (r"\s+and\s+", r"\s+y\s+"):
            match = re.search(connector, intent, re.IGNORECASE)
            if match:
                left = intent[: match.start()].strip()
                right = intent[match.end() :].strip()
                if len(left.split()) >= 2 and len(right.split()) >= 2:
                    return [left, right]
        return [intent]

    # ─── Layer 1 ──────────────────────────────────────────────
    def normalize_surface(self, intent: str) -> str:
        n = intent
        # Special case: bare "ven/vení/come here" without specific target
        bare_come = re.compile(
            r"^\s*(ven[ií]?|venite|come)\s*"
            r"(aca|acá|aqui|aquí|here|por aqui|por aca)?"
            r"[\s,\.!?]*$",
            re.IGNORECASE,
        )
        bare_approach = re.compile(
            r"^\s*acerc[aá]te[\s,\.!?]*$",
            re.IGNORECASE,
        )
        s = n.strip()
        if bare_come.match(s) or bare_approach.match(s):
            return f"Follow the player named {self.player_name} and stay within 3 blocks."
        # Verb mapping
        for es, en in self.VERB_MAP:
            n = re.sub(rf"\b{re.escape(es)}\b", en, n, flags=re.IGNORECASE)
        # Compact whitespace
        n = re.sub(r"\s+", " ", n).strip()
        # Pronoun replacements
        player_re = re.escape(self.player_name)
        n = re.sub(rf"\bdel jugador(?:\s+llamado)?\s+{player_re}\b", f"of the player named {self.player_name}", n, flags=re.IGNORECASE)
        n = re.sub(rf"\bal jugador(?:\s+llamado)?\s+{player_re}\b", f"to the player named {self.player_name}", n, flags=re.IGNORECASE)
        n = re.sub(rf"\bel jugador(?:\s+llamado)?\s+{player_re}\b", f"the player named {self.player_name}", n, flags=re.IGNORECASE)
        n = re.sub(r"\btu posición\b", "your current position", n, flags=re.IGNORECASE)
        n = re.sub(r"\bal jugador\b", f"to the player named {self.player_name}", n, flags=re.IGNORECASE)
        n = re.sub(r"\bel jugador\b", f"the player named {self.player_name}", n, flags=re.IGNORECASE)
        n = re.sub(r"\bdel jugador\b", f"of the player named {self.player_name}", n, flags=re.IGNORECASE)
        # Ensure terminal period
        if not n.endswith("."):
            n = n + "."
        # Capitalize first letter
        if n and n[0].islower():
            n = n[0].upper() + n[1:]
        return n

    # ─── Orchestrator ─────────────────────────────────────────
    def execute(self, user_intent: str) -> dict:
        """Run the full L2→L3→L5→(L1+L4) pipeline.

        Returns:
            dict with keys:
              - ok (bool)
              - outcome (str): "policy_handled_upstream" or "embodied_ready"
              - policy_handled (bool)
              - policy_layer (str|None): "scope" or "ambiguity" if cut
              - policy_reason (str|None)
              - sub_intents (list[str]): normalized atomic intents
              - categories (list[str]): per sub-intent
              - allowed_tools (list[list[str]|None]): per sub-intent
              - execution_results (list): always [] (reserved for downstream)
              - plan (None): reserved for downstream
        """
        # L2 — scope
        oos, reason = self.is_out_of_scope(user_intent)
        if oos:
            return {
                "ok": True,
                "outcome": "policy_handled_upstream",
                "policy_handled": True,
                "policy_layer": "scope",
                "policy_reason": f"out_of_scope: matched '{reason}'",
                "sub_intents": [],
                "categories": [],
                "allowed_tools": [],
                "execution_results": [],
                "plan": None,
            }

        # L3 — ambiguity
        amb, token = self.is_ambiguous(user_intent)
        if amb:
            return {
                "ok": True,
                "outcome": "policy_handled_upstream",
                "policy_handled": True,
                "policy_layer": "ambiguity",
                "policy_reason": f"ambiguous: matched '{token}'; ask user for clarification",
                "sub_intents": [],
                "categories": [],
                "allowed_tools": [],
                "execution_results": [],
                "plan": None,
            }

        # L5 — decompose
        raw_subs = self.decompose(user_intent)

        # L1 + L4 per sub-intent
        normalized_chain: list[str] = []
        category_chain: list[str] = []
        allowed_chain: list[list[str] | None] = []

        for sub in raw_subs:
            normalized = self.normalize_surface(sub)
            normalized_chain.append(normalized)
            category = self.classify_category(normalized)
            category_chain.append(category)
            allowed = self.get_allowed_tools(category)
            allowed_chain.append(allowed)

        return {
            "ok": True,
            "outcome": "embodied_ready",
            "policy_handled": False,
            "policy_layer": None,
            "policy_reason": None,
            "sub_intents": normalized_chain,
            "categories": category_chain,
            "allowed_tools": allowed_chain,
            "execution_results": [],
            "plan": None,
        }
