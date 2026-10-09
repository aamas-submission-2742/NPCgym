from collections.abc import Iterable
from typing import Protocol, cast

import gymnasium as gym


class _TaxiDecoder(Protocol):
    def decode(self, state: int) -> Iterable[int]: ...


class IgnoreWeatherRelevant(gym.ObservationWrapper[int, int, int]):
    """Compress Storm Taxi states by dropping norm-only fields: home, flood, shelter."""

    def __init__(self, env: gym.Env[int, int]) -> None:
        super().__init__(env)
        # (row, col, passenger, destination, rain, hurricane)
        self.observation_space = gym.spaces.Discrete(5 * 5 * 5 * 4 * 2 * 11)

    def _encode_compact(self, row: int, col: int, passenger: int, destination: int, rain: int, hurricane: int) -> int:
        s = row
        s = s * 5 + col
        s = s * 5 + passenger
        s = s * 4 + destination
        s = s * 2 + int(rain)
        s = s * 11 + hurricane
        return s

    def observation(self, observation: int) -> int:
        base_env = cast(_TaxiDecoder, self.env.unwrapped)
        row, col, passenger, destination, rain, hurricane, _, _, _ = (
            int(value) for value in base_env.decode(int(observation))
        )
        return self._encode_compact(row, col, passenger, destination, rain, hurricane)
