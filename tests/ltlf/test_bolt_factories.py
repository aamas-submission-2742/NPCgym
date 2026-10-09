import pytest

from npc_gym.bolts import make_ltlf_bolt, make_regex_bolt
from npc_gym.envs import StormTaxiEnv
from npc_gym.wrappers import RestrainingBoltWrapper


@pytest.mark.parametrize("reporting", ["prefix", "restart"])
@pytest.mark.parametrize("consume_initial", [False, True])
def test_real_ltlf_and_regex_bolts_match_rewards(compiler, reporting, consume_initial):
    options = {
        "propositions": {"a": lambda value: True},
        "reward": -2,
        "reporting": reporting,
        "consume_initial": consume_initial,
    }
    specs = [make_ltlf_bolt("G a", compiler=compiler, **options), make_regex_bolt("[a]*", **options)]
    envs = [RestrainingBoltWrapper(StormTaxiEnv(), bolts={"a": spec}) for spec in specs]
    try:
        reset_infos = [env.reset(seed=3)[1]["restraining_bolts"] for env in envs]
        assert reset_infos[0]["occurrences"] == reset_infos[1]["occurrences"]
        for _ in range(3):
            results = [env.step(0) for env in envs]
            assert results[0][1] == results[1][1]
            assert (
                results[0][4]["restraining_bolts"]["occurrences"] == results[1][4]["restraining_bolts"]["occurrences"]
            )
            assert results[0][4]["restraining_bolts"]["reward_adjustments"] == {"a": -2}
    finally:
        for env in envs:
            env.close()


def test_default_ltlf_bolt_compiler(compiler):
    # The fixture requires real optional dependencies; exercise the default path too.
    bolt = make_ltlf_bolt("a", propositions={"a": lambda value: True}, reward=1)
    assert bolt.definition.atoms == ("a",)
