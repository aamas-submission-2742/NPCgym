"""Storm Taxi composed with Gymnasium's dry Taxi dynamics.

The state/RNG contract and weather rules preserve NPC Gym's original storm
variant. See THIRD_PARTY_NOTICES.md for Gymnasium and retained-code attribution.
"""

from collections.abc import Iterator
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import ResetNeeded
from numpy.typing import NDArray

from npc_gym.envs.metrics import EPISODE_METRICS_KEY
from npc_gym.envs.taxi._storm_rendering import StormRenderer
from npc_gym.envs.taxi._storm_weather import TransitionDistribution, weather_outcomes
from npc_gym.envs.taxi._taxi_backend import TaxiBackend
from npc_gym.envs.taxi.labels import TaxiAuthorityState, TaxiLabelingFunction
from npc_gym.labels import Transition


class StormTaxiEnv(gym.Env[int, int]):
    """Deliver a passenger while weather creates warning and shelter obligations.

    Observations are ``Discrete(352000)`` integers encoding row, column,
    passenger, destination, rain, hurricane stage, home, flood risk and shelter,
    with radices (5, 5, 5, 4, 2, 11, 4, 2, 4). Passenger 4 is inside the taxi.
    Actions 0..5 are Gymnasium's south, north, east, west, pickup and dropoff;
    action 6 warns without moving. Movement stays deterministic during rain.

    Episodes start clear. Weather evolves after each action. Clear-weather and
    active-hurricane steps reward -1, even on delivery. Rain without a hurricane
    retains upstream rewards: -1 normally, -10 for illegal pickup/dropoff, +20
    for delivery. Delivery always terminates; direct construction has no limit.

    ``fickle_passenger=True`` gives each episode a 30% chance of one destination
    change on the first successful movement with a passenger already aboard.
    Reset and step information contains ``prob`` (the sampled branch probability),
    a detached advisory ``action_mask``, frozen ``labels`` and delivery ``success``
    under ``episode_metrics``. Location labels describe actual arrival and progress.

    ``render_mode`` selects headless ``None``, ``ansi``, ``rgb_array`` or ``human``.
    Graphical modes overlay storm information on Gymnasium frames and require
    the render extra. ``npc_gym/StormTaxi-v0`` supplies a 50-step limit.
    """

    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}  # noqa: RUF012 - Gymnasium metadata contract.
    action_space: spaces.Discrete[np.int64]  # type: ignore[assignment]
    observation_space: spaces.Discrete[np.int64]  # type: ignore[assignment]
    s: int

    def __init__(self, render_mode: str | None = None, fickle_passenger: bool = False) -> None:
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(
                f"Unsupported render_mode {render_mode!r}; expected one of {self.metadata['render_modes']} or None"
            )
        self.render_mode = render_mode
        self._backend = TaxiBackend("rgb_array" if render_mode == "human" else render_mode)
        self._renderer: StormRenderer | None = None
        self.locs = self._backend.locations
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Discrete(352000)
        self.labeling_function = TaxiLabelingFunction()
        self.fickle_passenger = fickle_passenger
        self.fickle_step = self.fickle_passenger and self.np_random.random() < 0.3
        self.lastaction: int | None = None
        self.is_raining = False
        self.hurricane = 0
        self.home: int | None = None
        self.flood = False
        self.shelter: int | None = None
        self._labeling_state_ready = False
        self.initial_state_distrib: NDArray[np.float64] = np.zeros(352000)
        for base_state in np.flatnonzero(self._backend.initial_distribution):
            passenger, destination = divmod(int(base_state) % 20, 4)
            for home in range(4):
                for flood in range(2):
                    for shelter in range(4):
                        if shelter not in (passenger, destination, home):
                            self.initial_state_distrib[int(base_state) * 704 + home * 8 + flood * 4 + shelter] = 1
        self.initial_state_distrib /= self.initial_state_distrib.sum()

    def encode(
        self,
        taxi_row: int,
        taxi_col: int,
        pass_loc: int,
        dest_idx: int,
        rain: bool | int | None = None,
        hurricane: int | None = None,
        home: int | None = None,
        flood: bool | int | None = None,
        shelter: int | None = None,
    ) -> int:
        """Encode nine fields; omitted storm fields fall back to instance attributes.

        Missing home/shelter attributes raise TypeError, including before reset.
        Reset/step state is read through ``decode(s)`` or ``labeling_state()``;
        the fallback attributes are explicit encoding defaults, not a state mirror.
        """
        rain = self.is_raining if rain is None else rain
        hurricane = self.hurricane if hurricane is None else hurricane
        home = self.home if home is None else home
        flood = self.flood if flood is None else flood
        shelter = self.shelter if shelter is None else shelter
        if home is None or shelter is None:
            raise TypeError("Storm Taxi encoding requires home and shelter indices")
        base = ((taxi_row * 5 + taxi_col) * 5 + pass_loc) * 4 + dest_idx
        return base * 704 + int(rain) * 352 + hurricane * 32 + home * 8 + int(flood) * 4 + shelter

    def decode(self, state: int) -> Iterator[int]:
        """Return a single-use iterator over the nine encoded components."""
        fields = []
        for radix in (4, 2, 4, 11, 2, 4, 5, 5):
            state, value = divmod(state, radix)
            fields.append(value)
        assert 0 <= state < 5
        fields.append(state)
        return reversed(fields)

    def get_state(self) -> int:
        """Return the current encoded observation after reset or step."""
        return self.s

    def labeling_state(self) -> TaxiAuthorityState:
        """Return an immutable authority snapshot, detached from mutable state."""
        if not self._labeling_state_ready:
            raise ResetNeeded("StormTaxiEnv must be reset before labeling_state()")
        row, column, passenger, destination, rain, hurricane, home, flood, shelter = self.decode(int(self.s))
        return TaxiAuthorityState(
            taxi_position=(column, row),
            passenger=passenger,
            destination=destination,
            raining=bool(rain),
            hurricane=hurricane,
            home=home,
            flood_risk=bool(flood),
            shelter=shelter,
            locations=tuple((column, row) for row, column in self.locs),
        )

    def action_mask(self, state: int | None = None) -> NDArray[np.int8]:
        """Return a fresh seven-action advisory mask; Warn is always available."""
        if state is None:
            state = self.s
        if not self.observation_space.contains(state):
            raise ValueError(f"State {state!r} is outside {self.observation_space}")
        return cast(NDArray[np.int8], self._backend.masks[int(state) // 704].copy())

    def transition_distribution(self, state: int, action: int) -> TransitionDistribution:
        """Return ordered weather/action outcomes without sampling or mutating state.

        This distribution excludes the history-dependent fickle-passenger change,
        which is applied by ``step`` after sampling, as in the original variant.
        """
        if not self.observation_space.contains(state):
            raise ValueError(f"State {state!r} is outside {self.observation_space}")
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside {self.action_space}")
        return self._distribution(int(state), int(action))

    def _distribution(self, state: int, action: int) -> TransitionDistribution:
        base, weather = divmod(state, 704)
        _, successor, reward, terminated = self._backend.outcome(base, action)
        return weather_outcomes(successor, weather, action, reward, terminated)

    def step(self, action: int | np.int64) -> tuple[int, int, bool, bool, dict[str, Any]]:
        previous = int(self.s)
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside {self.action_space}")
        action = int(action)
        outcomes = self._distribution(previous, action)
        index = categorical_sample([outcome[0] for outcome in outcomes], self.np_random)
        probability, state, reward, terminated = outcomes[index]
        self.lastaction = action
        if self.fickle_passenger and self.fickle_step:
            old_base, new_base = previous // 704, state // 704
            # Each physical cell has 20 passenger/destination combinations.
            if old_base // 4 % 5 == 4 and old_base // 20 != new_base // 20:
                self.fickle_step = False
                destination = int(self.np_random.choice([value for value in range(4) if value != old_base % 4]))
                state += (destination - new_base % 4) * 704
        self.s = state
        if self.render_mode == "human":
            self.render()
        return state, reward, terminated, False, self._info(previous, action, probability, terminated)

    def _info(self, previous: int | None, action: int | None, probability: float, terminated: bool) -> dict[str, Any]:
        return {
            "prob": probability,
            "action_mask": self._backend.masks[self.s // 704].copy(),
            "labels": self.labeling_function(Transition(previous, action, self.s, terminated, False)),
            EPISODE_METRICS_KEY: {"success": int(terminated)},
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        self.s = int(categorical_sample(self.initial_state_distrib, self.np_random))
        self._labeling_state_ready = True
        self.lastaction = None
        self.fickle_step = self.fickle_passenger and self.np_random.random() < 0.3
        self._backend.reset_orientation()
        if self.render_mode == "human":
            self.render()
        return self.s, self._info(None, None, 1.0, False)

    def render(self) -> str | NDArray[np.uint8] | None:
        """Render upstream Taxi plus storm overlays; pace human frames only."""
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render() without specifying render_mode at construction; no image will be produced."
            )
            return None
        if not self._labeling_state_ready:
            raise ResetNeeded("StormTaxiEnv must be reset before rendering")
        if self.render_mode == "ansi":
            return self._backend.render(int(self.s) // 704, self.lastaction)
        if self._renderer is None:
            self._renderer = StormRenderer(self._backend)
        try:
            return self._renderer.render(
                int(self.s),
                self.lastaction,
                self.labeling_state(),
                human=self.render_mode == "human",
                fps=self.metadata["render_fps"],
            )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release rendering resources; the environment may be reset and reused."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
