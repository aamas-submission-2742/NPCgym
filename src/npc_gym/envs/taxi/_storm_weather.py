"""NPC Gym weather composition; ordinary Taxi physics come from Gymnasium."""

from functools import lru_cache

TransitionOutcome = tuple[float, int, int, bool]
TransitionDistribution = tuple[TransitionOutcome, ...]


@lru_cache(maxsize=4096)
def weather_outcomes(
    base_successor: int, weather: int, action: int, reward: int, terminated: bool
) -> TransitionDistribution:
    """Attach weather to a base outcome, retaining order and historical rewards.

    ``weather`` is the low 704 values of the storm encoding. The low 32 values
    (home, flood, shelter) never change; rain and hurricane occupy the rest.
    The cache contains values only, never environment instances or RNG state.
    """
    rain, rest = divmod(weather, 352)
    hurricane, locations = divmod(rest, 32)
    base = base_successor * 704
    intended = base + weather
    if not rain:
        return (
            (0.75, intended, -1, terminated),
            (0.2, intended + 352, -1, terminated),
            (0.05, base + 352 + 32 + locations, -1, terminated),
        )
    if hurricane == 0:
        stay, start = (0.8, 0.2) if action < 4 else (0.9, 0.1)
        return ((stay, intended, reward, terminated), (start, intended + 32, reward, terminated))
    next_hurricane = 0 if hurricane == 10 else hurricane + 1
    return ((1.0, base + 352 + next_hurricane * 32 + locations, -1, terminated),)
