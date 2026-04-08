"""Registry of prompt version hashes, computed once at import time.

Each hash is a 12-character hex prefix of the SHA-256 digest of the
prompt text.  When a prompt changes, the hash changes — this is enough
to detect prompt drift across experiment runs without storing the full
prompt text with every variant.
"""

from __future__ import annotations

import hashlib


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# Import all prompt constants
from app.workflow.agents.ideator import SYSTEM_PROMPT as _IDEATOR
from app.workflow.agents.ideator import REPLACEMENT_PROMPT as _IDEATOR_REPLACE
from app.workflow.agents.idea_critic import SYSTEM_PROMPT as _IDEA_CRITIC
from app.workflow.agents.critic import SYSTEM_PROMPT as _CRITIC
from app.workflow.agents.critic import BLIND_COMPARISON_PROMPT as _BLIND
from app.workflow.agents.refiner import SYSTEM_PROMPT as _REFINER

PROMPT_VERSIONS: dict[str, str] = {
    "ideator": _hash(_IDEATOR),
    "ideator_replacement": _hash(_IDEATOR_REPLACE),
    "idea_critic": _hash(_IDEA_CRITIC),
    "critic": _hash(_CRITIC),
    "critic_blind": _hash(_BLIND),
    "refiner": _hash(_REFINER),
}
