"""One-step Environment Friendly policy fixing for Merchant."""

from importlib.resources import files

from npc_gym.envs.merchant.labels import MerchantAuthorityState
from npc_gym.policy_fixes.core import PlanningProblem, integer


class MerchantModel:
    """Predict Environment Friendly violations from the current snapshot.

    Only the current cell and whether any wood is carried affect the cost:
    Extract (4) costs one at an intact tree while carrying wood, even when
    inventory is full. Every other action costs zero. The horizon is one;
    movement, rewards and future inventory changes are not modeled.

    The model retains no environment or monitor history and can be reused
    across resets and layouts. Use ``ASPPlanner()`` to prioritize avoiding
    violations, then the policy's action preferences.
    """

    def __init__(self) -> None:
        self._rules = files("npc_gym.policy_fixes").joinpath("merchant.lp").read_text(encoding="utf-8")

    def problem(self, state: MerchantAuthorityState) -> PlanningProblem:
        """Build a one-step problem from ``env.labeling_state()``.

        Supply values for all seven actions to the planner. Only ``cell`` and
        ``carried_wood`` are validated and used; other snapshot fields are
        irrelevant to this norm. Call before episode end, which the snapshot
        does not identify. No history updates or reset notifications are needed.
        """
        if not isinstance(state, MerchantAuthorityState):
            raise TypeError("state must be a MerchantAuthorityState from labeling_state()")
        if not isinstance(state.cell, str) or state.cell not in {"H", "D", "M", "R", "T", "C", "."}:
            raise ValueError("cell must be a Merchant cell label: H, D, M, R, T, C or .")
        wood = integer("carried_wood", state.carried_wood)
        atoms = []
        if state.cell == "T":
            atoms.append("at_tree")
        if wood > 0:
            atoms.append("has_wood")
        return PlanningProblem.from_externals(static=self._rules, true_atoms=atoms)


__all__ = ["MerchantModel"]
