"""One-step Warn policy fixing for Storm Taxi."""

from importlib.resources import files

from npc_gym.envs.taxi.labels import TaxiWeatherLabel
from npc_gym.monitors.contracts import MonitorInput
from npc_gym.policy_fixes.core import PlanningProblem


class TaxiModel:
    """Require Warn (6) on the step immediately after each rain onset.

    Only rain history matters: an onset is a rainy step following a dry step,
    or a rainy first step. The one-step cost is one for any non-Warn action
    when a response is due, and zero otherwise. Other Emergency components,
    environment dynamics and rewards are not modeled.

    Use with Storm Taxi's seven actions. Construct or call ``reset`` after
    each environment reset, then ``advance`` once per actual transition.
    Reset observations are ignored, matching the monitor. Use one model per
    environment; the planner may reuse its grounded solver across resets.
    """

    def __init__(self) -> None:
        self._rules = files("npc_gym.policy_fixes").joinpath("taxi.lp").read_text(encoding="utf-8")
        self.reset()

    def reset(self) -> None:
        """Clear rain history and any pending response for a new episode."""
        self._raining = False
        self._warning_due = False
        self._ended = False

    def advance(self, input: MonitorInput) -> None:
        """Record one actual step, including its termination/truncation flags.

        Do not pass reset labels or hypothetical transitions. A pending response
        expires on this step regardless of its action; persistent rain does not
        rearm it. A later rain onset creates a new response obligation.
        """
        if not isinstance(input, MonitorInput):
            raise TypeError("advance requires a MonitorInput for one real transition")
        if self._ended:
            raise ValueError("Episode has ended; reset the model before advancing")
        raining = TaxiWeatherLabel.RAIN in input.labels
        self._warning_due = raining and not self._raining
        self._raining = raining
        self._ended = input.terminated or input.truncated

    def problem(self) -> PlanningProblem:
        """Build a one-step problem without changing history.

        Supply values for actions 0..6 to the planner. ``ASPPlanner()`` minimizes
        warning violations before policy preference. Excluding Warn through an
        action mask makes a due violation unavoidable. Calls after an observed
        episode ending are rejected; ending creates no additional norm event.
        """
        if self._ended:
            raise ValueError("Episode has ended; reset the model before planning")
        return PlanningProblem.from_externals(
            static=self._rules, true_atoms=["warning_due"] if self._warning_due else []
        )


__all__ = ["TaxiModel"]
