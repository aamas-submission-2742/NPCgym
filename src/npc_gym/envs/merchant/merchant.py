from __future__ import annotations

from importlib import resources
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium import Env
from gymnasium.error import DependencyNotInstalled, ResetNeeded
from numpy.typing import NDArray

from npc_gym.envs.merchant.labels import MerchantAuthorityState, MerchantLabelingFunction, MerchantState
from npc_gym.envs.metrics import EPISODE_METRICS_KEY
from npc_gym.labels import LabelSet, Transition

if TYPE_CHECKING:
    from npc_gym.envs.merchant.py_game_merchant_display import MerchantGraphicsPyGame

actions = ["north", "south", "east", "west", "extract", "unload", "fight"]
action_dict = {name: i for i, name in enumerate(actions)}

labels = ["H", "D", "M", "R", "T", "C", "."]
labels_dict = {name: i for i, name in enumerate(labels)}

directions: dict[str, NDArray[np.int64]] = {
    "north": np.array([0, -1]),
    "south": np.array([0, 1]),
    "east": np.array([1, 0]),
    "west": np.array([-1, 0]),
}


class MerchantEnv(Env[MerchantState, int]):
    # Gymnasium declares metadata as an instance attribute.
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}  # noqa: RUF012

    # Discrete accepts Python ints but samples NumPy ints; Gymnasium's invariant
    # Space type cannot express both while preserving the integer action API.
    action_space: gym.spaces.Discrete[np.int64]  # type: ignore[assignment]
    observation_space: gym.spaces.Tuple

    def __init__(
        self,
        layout: str = "basic",
        risk_fight: float = 1.0,
        risk_death: float = 0.0,
        capacity: int = 5,
        sunset: int = 28,
        render_mode: str | None = None,
        step_delay_ms: int = 0,
    ) -> None:
        self._validate_parameters(layout, risk_fight, risk_death, capacity, sunset, step_delay_ms)
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(
                f"Unsupported render_mode {render_mode!r}; expected one of {self.metadata['render_modes']} or None"
            )
        # set environment parameters
        self.layout = layout
        self.risk_fight = risk_fight
        self.risk_death = risk_death
        self.capacity = capacity
        self.sunset = sunset
        self.labeling_function = MerchantLabelingFunction(sunset)
        # the state reported by the previous transition, passed on as the next transition's origin
        self._last_state: MerchantState | None = None
        self.render_mode = render_mode
        self.render_step_delay_ms = step_delay_ms
        self.display: MerchantGraphicsPyGame | None = None
        self.load_map()
        self.wood_positions: dict[tuple[int | np.intp, int | np.intp], int] = {
            tuple(pos): i for i, pos in enumerate(np.argwhere(self.map == "T"))
        }
        self.ore_positions: dict[tuple[int | np.intp, int | np.intp], int] = {
            tuple(pos): i for i, pos in enumerate(np.argwhere(self.map == "R"))
        }
        self.max_carried_wood = min(self.capacity, len(self.wood_positions))
        self.max_carried_ore = min(self.capacity, len(self.ore_positions))
        # setup all state variables
        # set action space
        self.action_space = gym.spaces.Discrete(len(action_dict))
        # set observation space
        self.observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(self.map.shape[1]),  # x in [0, width-1], 0
                gym.spaces.Discrete(self.map.shape[0]),  # y in [0, height-1], 1
                gym.spaces.Discrete(len(labels)),  # label indicating what is on the current grid cell
                gym.spaces.Discrete(self.max_carried_wood + 1),  # carried wood count, 3
                gym.spaces.Discrete(self.max_carried_ore + 1),  # carried ore count, 4
                gym.spaces.Discrete(self.sunset + 1),  # clock that counts up till sunset, 5
                gym.spaces.Discrete(len(actions) + 1),  # last performed action
            )
            + (gym.spaces.Discrete(2),) * len(self.wood_positions)
            + (gym.spaces.Discrete(2),) * len(self.ore_positions)
        )
        self.reset()
        self._labeling_state_ready = False

    @staticmethod
    def _validate_parameters(
        layout: str, risk_fight: float, risk_death: float, capacity: int, sunset: int, step_delay_ms: int
    ) -> None:
        if not isinstance(layout, str) or not layout:
            raise ValueError("layout must be a non-empty string naming a bundled Merchant layout")
        for name, value in (("risk_fight", risk_fight), ("risk_death", risk_death)):
            if isinstance(value, bool) or not isinstance(value, Real) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a number between 0 and 1 inclusive; got {value!r}")
        if isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity < 1:
            raise ValueError(f"capacity must be a positive integer; got {capacity!r}")
        if isinstance(sunset, bool) or not isinstance(sunset, Integral) or sunset < 0:
            raise ValueError(f"sunset must be a non-negative integer; got {sunset!r}")
        if isinstance(step_delay_ms, bool) or not isinstance(step_delay_ms, Integral) or step_delay_ms < 0:
            raise ValueError(f"step_delay_ms must be a non-negative integer; got {step_delay_ms!r}")

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[MerchantState, dict[str, Any]]:
        super().reset(seed=seed)
        self.pos = self.home
        self.label: str | np.str_ = "H"
        self.carried_wood, self.carried_ore = 0, 0
        self.wood = [1 for _ in self.wood_positions]
        self.ore = [1 for _ in self.ore_positions]
        self.clock = 0  # clock counts up from 0 until sundown, time then stops being tracked
        self.action: str | None = None
        self._episode_score = 0
        self._unload_danger = 0
        self._unload_market = 0
        self._death = 0
        if self.render_mode == "human":
            self.render()
        state = self.get_state()
        self._last_state = state
        self._labeling_state_ready = True
        transition_labels = self.labeling_function(
            Transition(previous_state=None, action=None, state=state, terminated=False, truncated=False)
        )
        return state, self._get_info(transition_labels)

    def _get_obs(self) -> MerchantState:
        return (
            int(self.pos[0]),
            int(self.pos[1]),
            labels_dict[self.label],
            self.carried_wood,
            self.carried_ore,
            self.clock,
            action_dict[self.action] if self.action is not None else len(actions),
            *self.wood,
            *self.ore,
        )

    def _get_info(self, transition_labels: LabelSet) -> dict[str, Any]:
        return {
            "action_mask": self.action_mask(),
            "labels": transition_labels,
            EPISODE_METRICS_KEY: {
                "score": self._episode_score,
                "unload_danger": self._unload_danger,
                "unload_market": self._unload_market,
                "death": self._death,
            },
        }

    def load_map(self) -> None:
        """Load and validate the named bundled layout and its fixed locations."""
        layouts = resources.files("npc_gym.envs.merchant").joinpath("merchant_layouts")
        available = sorted(
            path.name.removesuffix(".txt")
            for path in layouts.iterdir()
            if path.name.endswith(".txt") and path.is_file()
        )
        if self.layout not in available:
            raise ValueError(f"Unknown Merchant layout {self.layout!r}; expected one of {available}")

        mapfile = layouts.joinpath(f"{self.layout}.txt")
        rows = mapfile.read_text(encoding="utf-8").splitlines()
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError(f"Merchant layout {self.layout!r} must be a non-empty rectangular grid")
        unknown = sorted(set().union(*map(set, rows)) - {*labels, "X"})
        if unknown:
            raise ValueError(f"Merchant layout {self.layout!r} contains unsupported cells: {unknown}")
        if sum(row.count("H") for row in rows) != 1 or sum(row.count("M") for row in rows) != 1:
            raise ValueError(f"Merchant layout {self.layout!r} must contain exactly one home 'H' and one market 'M'")
        if any(cell != "X" for cell in (*rows[0], *rows[-1])) or any(row[0] != "X" or row[-1] != "X" for row in rows):
            raise ValueError(f"Merchant layout {self.layout!r} must have an outer wall of 'X' cells")

        self.map: NDArray[np.str_] = np.asarray([list(row) for row in rows], dtype="U1")
        home_y, home_x = np.argwhere(self.map == "H")[0]
        market_y, market_x = np.argwhere(self.map == "M")[0]
        self.home: NDArray[np.intp] = np.array([home_x, home_y])
        self.market: NDArray[np.intp] = np.array([market_x, market_y])

    def _excluded_actions(self) -> list[int]:
        # this should be based purely on observations, so it can be determined by the learning agent
        observation = self._get_obs()
        x, y, label_index, carried_wood, carried_ore, _, action = observation[:7]
        result: set[int] = set()
        label = labels[label_index]
        # we are not allowed to run into a wall
        for a, d in directions.items():
            target = np.array([x, y]) + d
            if self.map[target[1], target[0]] == "X":
                result.add(action_dict[a])
        # we are only allowed to collect on an appropriate cell given we have capacity
        if label not in ["T", "R"] or (carried_wood + carried_ore >= self.capacity):
            result.add(action_dict["extract"])
        # we are only allowed to unload if we are attacked / at the market and have something to unload
        if (carried_wood + carried_ore == 0) or label not in ["D", "M"]:
            result.add(action_dict["unload"])
        # we are allowed to fight only if we are attacked
        if label != "D":
            result.add(action_dict["fight"])
        # if we are attacked we are only allowed to fight or unload
        if label == "D":
            result |= {action_dict[a] for a in ["north", "south", "east", "west", "extract"]}
        # Forbid reversal only if another action remains available. Other restrictions always hold.
        if action < len(actions) and actions[action] in directions:
            for a, d in directions.items():
                if all(d == -directions[actions[action]]):
                    excluded_with_reversal = result | {action_dict[a]}
                    if len(excluded_with_reversal) < len(actions):
                        result = excluded_with_reversal
        return list(result)

    def action_mask(self) -> NDArray[np.int8]:
        """Return an advisory mask, allowing reversal only when no other valid action remains.

        Wall, danger, resource, and inventory restrictions always apply. The
        no-reversal rule never removes the last otherwise available action.
        """
        mask = np.ones(len(actions), dtype=np.int8)
        mask[self._excluded_actions()] = 0
        return mask

    def _proceed(self, reward: int, terminal: bool) -> tuple[MerchantState, int, bool, bool, dict[str, Any]]:
        # Every step branch records its action before completing the transition.
        assert self.action is not None
        # remembers the current reward
        self.reward = reward
        self._episode_score += reward
        # refresh visualization on every state transition
        if self.render_mode in ["human", "rgb_array"]:
            self.render()
        state = self.get_state()
        previous_state, self._last_state = self._last_state, state
        transition_labels = self.labeling_function(
            Transition(
                previous_state=previous_state,
                action=action_dict[self.action],
                state=state,
                terminated=terminal,
                truncated=False,
            )
        )
        return (
            state,
            self.reward,
            terminal,
            False,
            self._get_info(transition_labels),
        )

    def step(self, action: int | np.int64) -> tuple[MerchantState, int, bool, bool, dict[str, Any]]:
        """Advance time and apply the action; extraction at capacity is a zero-reward no-op."""
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside {self.action_space}")
        action = int(action)
        # we use the action string
        self.action = actions[action]
        # increase clock, stop at sunset (afterwards we don't have to keep track of time anymore)
        if self.clock < self.sunset:
            self.clock += 1
        # if the agent is on a danger cell, they will directly be attacked
        # in this case, they can unload or try to fight
        if self.label == "D":
            # unloading will always work
            if self.action == "unload":
                carried = self.carried_wood + self.carried_ore
                self.label = "."  # no more danger (for now, if agent re-visits the field the danger is back)
                self.carried_ore, self.carried_wood = 0, 0  # but also no more resources
                self._unload_danger += 1
                return self._proceed(-50 * carried, False)  # agent has to give back the 50 per resource
            # fighting will often lead to death and sometimes work
            elif self.action == "fight":
                if self.np_random.random() <= self.risk_death:
                    self._death = 1
                    return self._proceed(-100, True)  # agent dies with -100 reward, episode terminates
                else:
                    self.label = "."  # no more danger (for now, if agent re-visits the field the danger is back)
                    return self._proceed(0, False)
            else:
                # all other actions have no further effect (except progressing time)
                return self._proceed(0, False)
        # if agent is not attacked, they can walk around, collect, and unload
        else:
            # the agent can simply move around
            if self.action in ["north", "east", "south", "west"]:
                newpos = self.pos + directions[self.action]
                if (
                    (0 <= newpos[0] < self.map.shape[1])
                    and (0 <= newpos[1] < self.map.shape[0])
                    and self.map[newpos[1], newpos[0]] != "X"
                ):
                    self.pos = newpos
                    # we set the label so the agent can observe what is at their new position
                    self.label = self.map[self.pos[1], self.pos[0]]
                    if self.label == "T" and self.wood[self.wood_positions[(self.pos[1], self.pos[0])]] == 0:
                        self.label = "C"  # wood already collected
                    if self.label == "R" and self.ore[self.ore_positions[(self.pos[1], self.pos[0])]] == 0:
                        self.label = "C"  # ore already collected
                    if self.label == "D" and self.np_random.random() >= self.risk_fight:
                        self.label = "."  # we got lucky: no fight!
                return self._proceed(0, False)
            # the agent can extract a resource
            elif self.action == "extract":
                if self.carried_wood + self.carried_ore >= self.capacity:
                    return self._proceed(0, False)
                if self.label == "T":  # extracting trees
                    self.carried_wood += 1
                    self.wood[self.wood_positions[(self.pos[1], self.pos[0])]] = 0  # has now been collected
                    self.label = "C"
                    return self._proceed(50, False)  # the agent gains 50
                elif self.label == "R":  # extracting ore
                    self.carried_ore += 1
                    self.ore[self.ore_positions[(self.pos[1], self.pos[0])]] = 0  # has now been collected
                    self.label = "C"
                    return self._proceed(50, False)  # the agent gains 50
                else:  # extracting has no effect if there is nothing to extract
                    return self._proceed(0, False)
            # the agent can unload at the goal or somewhere else
            elif self.action == "unload":
                carried = self.carried_wood + self.carried_ore
                self.carried_wood, self.carried_ore = 0, 0
                if self.label == "M":  # unloading at the market ends the episode with 100 reward per resource
                    self._unload_market = 1
                    return self._proceed(100 * carried, True)  # episode is terminal
                else:  # unloading somewhere else loses the resources and 50 reward per resource
                    return self._proceed(-50 * carried, False)
            # all other actions (there is only "fight") have no effects
            else:
                return self._proceed(0, False)

    def render(self) -> NDArray[np.uint8] | None:
        """Draw using the constructor-selected mode; delay only human frames."""
        if self.render_mode not in ["human", "rgb_array"]:
            return None
        if self.display is None:
            try:
                from .py_game_merchant_display import MerchantGraphicsPyGame
            except ModuleNotFoundError as error:
                if error.name != "pygame":
                    raise
                raise DependencyNotInstalled(
                    'Merchant rendering requires pygame; install it with `pip install "npc-gym[render]"`'
                ) from error

            self.display = MerchantGraphicsPyGame(
                show_window=self.render_mode == "human", step_delay_ms=self.render_step_delay_ms
            )
        try:
            image = self.display.render(self)
        except BaseException:
            self.close()
            raise
        if self.render_mode == "rgb_array":
            return image
        return None

    def close(self) -> None:
        display, self.display = self.display, None
        if display is not None:
            display.close()

    def get_state(self) -> MerchantState:
        """Return position, cell, inventory counts, clock, last action, and resource flags.

        Resource flags list wood before ore, each in layout row-major order.
        Before the first step, the last-action entry is ``len(actions)``.
        """
        return self._get_obs()

    def labeling_state(self) -> MerchantAuthorityState:
        """Return a detached immutable snapshot for external labelers."""
        if not self._labeling_state_ready:
            raise ResetNeeded("MerchantEnv must be reset before labeling_state()")
        return MerchantAuthorityState(
            position=(int(self.pos[0]), int(self.pos[1])),
            cell=str(self.label),
            carried_wood=int(self.carried_wood),
            carried_ore=int(self.carried_ore),
            time=int(self.clock),
            wood_available=tuple(bool(value) for value in self.wood),
            ore_available=tuple(bool(value) for value in self.ore),
            layout_name=self.layout,
            layout=tuple("".join(row) for row in self.map),
        )
