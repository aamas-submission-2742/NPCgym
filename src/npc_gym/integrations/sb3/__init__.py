"""Stable-Baselines3 adapters, loaded only when their public names are used."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from npc_gym.integrations.sb3.callback import IntermediateEvaluation as IntermediateEvaluation
    from npc_gym.integrations.sb3.callback import SB3EvaluationCallback as SB3EvaluationCallback
    from npc_gym.integrations.sb3.callback import SB3Policy as SB3Policy
    from npc_gym.integrations.sb3.often import DQNOFTEN as DQNOFTEN

_PUBLIC_NAMES = frozenset({"DQNOFTEN", "IntermediateEvaluation", "SB3EvaluationCallback", "SB3Policy"})


def __getattr__(name: str) -> Any:
    if name not in _PUBLIC_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if find_spec("stable_baselines3") is None:
        raise ImportError(
            "Stable-Baselines3 integration requires the 'sb3' extra; install it with `pip install 'npc-gym[sb3]'`."
        )

    if name == "DQNOFTEN":
        from npc_gym.integrations.sb3.often import DQNOFTEN

        value = DQNOFTEN
    else:
        from npc_gym.integrations.sb3 import callback

        value = getattr(callback, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_PUBLIC_NAMES})


__all__ = ["DQNOFTEN", "IntermediateEvaluation", "SB3EvaluationCallback", "SB3Policy"]
