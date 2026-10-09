import gymnasium as gym

from npc_gym.envs.merchant.labels import MerchantState

MerchantObservation = MerchantState


class IgnoreTimeObservation(gym.ObservationWrapper[MerchantObservation, int, MerchantObservation]):
    """Drops the clock entry from Merchant observations."""

    def __init__(self, env: gym.Env[MerchantObservation, int]) -> None:
        super().__init__(env)
        if not isinstance(self.env.observation_space, gym.spaces.Tuple) or len(self.env.observation_space.spaces) < 7:
            raise TypeError("IgnoreTimeObservation requires a Merchant tuple observation with a clock field")
        spaces = self.env.observation_space.spaces
        self.observation_space = gym.spaces.Tuple(spaces[:5] + spaces[6:])

    def observation(self, observation: MerchantObservation) -> MerchantObservation:
        return observation[:5] + observation[6:]
