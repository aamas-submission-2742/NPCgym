"""Numerical and lifecycle contracts for two-stream tabular teaching."""

import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

from npc_gym.algorithms import TabularOFTEN, TabularQLearning, TeachingSession
from npc_gym.algorithms.tabular_q_learning import _state_key
from npc_gym.policy_fixes import ASPPlanner, PlanningProblem

pytest.importorskip("clingo")


class TinyEnv(gym.Env):
    def __init__(self, *, terminated=False, truncated=False, start=0, reward=1.0):
        self.action_space = Discrete(2, start=start)
        self.observation_space = Box(0, 100, (1,), dtype=np.int64)
        self.terminated = terminated
        self.truncated = truncated
        self.reward = reward
        self.actions = []
        self.seeds = []
        self.draws = []
        self.observation = np.zeros(1, dtype=np.int64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.seeds.append(seed)
        self.observation[:] = 0
        return self.observation, {"labels": frozenset(), "action_mask": [1, 1]}

    def step(self, action):
        self.actions.append(action)
        self.draws.append(self.np_random.random())
        self.observation += 1
        return (
            self.observation,
            self.reward,
            self.terminated,
            self.truncated,
            {
                "labels": frozenset(),
                "action_mask": [1, 1],
            },
        )


def fixture(*, env=None, expert=None, initial=(2.0, 0.0), learning_rate=0.1, **kwargs):
    env = TinyEnv() if env is None else env
    expert = TinyEnv() if expert is None else expert
    policy = TabularQLearning(env, learning_rate=learning_rate, gamma=0.5, log_interval=None)
    policy._q[_state_key(np.array([0]))] = np.array(initial, dtype=float)
    sessions, inputs = [], []
    start = policy.action_start
    problem = PlanningProblem(f'candidate(0,{start}..{start + 1}). cost("violations",1,harm) :- action(0,{start}).')

    def factory(observation, info):
        history = []
        inputs.append(history)
        session = TeachingSession(ASPPlanner(), lambda obs, data: problem, history.append)
        sessions.append(session)
        return session

    config = {"batch_size": 1, "epsilon": 0.0, "seed": 41}
    config.update(kwargs)
    trainer = TabularOFTEN(policy, expert, ordinary_session=factory, expert_session=factory, **config)
    return trainer, env, expert, sessions, inputs


def test_two_stream_filtering_budgets_history_and_unchanged_rewards():
    trainer, env, expert, sessions, inputs = fixture()
    steps = []
    trainer.learn(2, callback=steps.append)
    assert env.actions == [0] and expert.actions == [1]
    assert steps[0].decision.changed and not steps[0].retained
    assert steps[1].retained and steps[1].decision.changed
    assert steps[0].ordinary_td_loss == steps[0].expert_loss == 0
    assert len(sessions) == 2 and sessions[0] is not sessions[1]
    assert [len(items) for items in inputs] == [1, 1]
    assert trainer.stats.total_steps == 2
    assert trainer.stats.ordinary_steps == trainer.stats.expert_steps == 1
    assert trainer.stats.ordinary_return == trainer.stats.expert_return == 1.0
    assert trainer.stats.filtered_steps == 1
    assert trainer.stats.ordinary_buffer_size == 0 and trainer.stats.expert_buffer_size == 1
    assert trainer.stats.updates == 1
    assert trainer.policy.num_timesteps == 0
    assert env.seeds[0] != expert.seeds[0]
    # TD raises the demonstrated action; upstream's detached-target loss lowers
    # the maximum competing action. There is no gradient through the expert target.
    np.testing.assert_allclose(trainer.policy.q_values(np.array([0])), [1.9, 0.1])
    assert steps[1].expert_td_loss == 0.5
    assert steps[1].expert_loss == 2.8


@pytest.mark.parametrize("terminated,truncated,expected", [(True, False, 0.1), (False, True, 0.3), (True, True, 0.1)])
def test_td_termination_and_truncation(terminated, truncated, expected):
    trainer, *_ = fixture(
        env=TinyEnv(terminated=terminated, truncated=truncated),
        expert=TinyEnv(terminated=terminated, truncated=truncated),
        expert_weight=0,
    )
    trainer._target[_state_key(np.array([1]))] = np.array([4.0, 3.0])
    trainer.learn(2)
    assert trainer.policy.q_values(np.array([0]))[1] == pytest.approx(expected)


@pytest.mark.parametrize("error_sign,expected", [(1, [1.9, 0.0]), (-1, [2.1, 0.0]), (0, [2.0, 0.0])])
def test_absolute_loss_subgradient_and_detached_target(error_sign, expected):
    trainer, *_ = fixture(expert=TinyEnv(reward=0), margin=1)
    trainer._target[_state_key(np.array([0]))][1] = {1: 0.0, -1: 4.0, 0: 3.0}[error_sign]
    trainer.learn(2)
    np.testing.assert_allclose(trainer.policy.q_values(np.array([0])), expected)


def test_margin_maximum_tie_uses_lowest_action_index():
    trainer, *_ = fixture(initial=(0, 1), margin=1, expert=TinyEnv(reward=1))
    trainer.learn(2)
    # max(Q+margin)=1 ties, target expert=1 => exactly zero subgradient.
    np.testing.assert_array_equal(trainer.policy.q_values(np.array([0])), [0, 1])
    trainer, *_ = fixture(initial=(0, 1), margin=1, expert=TinyEnv(reward=1))
    trainer._target[_state_key(np.array([0]))][1] = 0
    trainer.learn(2)
    # Ordinary TD also runs (proposal already agrees); expert max tie lowers 0.
    np.testing.assert_allclose(trainer.policy.q_values(np.array([0])), [-0.1, 1])


def test_target_stays_frozen_until_update_interval():
    trainer, *_ = fixture(target_update_interval=2)
    initial = trainer._target[_state_key(np.array([0]))].copy()
    trainer.learn(2)
    np.testing.assert_array_equal(trainer._target[_state_key(np.array([0]))], initial)
    trainer.learn(3)
    assert trainer.stats.updates == 2
    np.testing.assert_array_equal(trainer._target[_state_key(np.array([0]))], trainer.policy.q_values(np.array([0])))
    trainer.policy._q[_state_key(np.array([0]))][0] = 99
    assert trainer._target[_state_key(np.array([0]))][0] != 99


@pytest.mark.parametrize("filter_replay,expert_weight", [(True, 0), (False, 0), (False, 1)])
def test_ablations(filter_replay, expert_weight):
    trainer, env, expert, *_ = fixture(filter_replay=filter_replay, expert_weight=expert_weight)
    trainer.learn(2)
    assert env.actions == [0] and expert.actions == [1]
    assert trainer.stats.ordinary_buffer_size == int(not filter_replay)
    assert trainer.stats.expert_buffer_size == 1
    # The expert TD term survives removing the expert margin loss.
    assert trainer.policy.q_values(np.array([0]))[1] > 0
    if filter_replay:
        assert trainer.policy.q_values(np.array([0]))[0] == 2


def test_empty_buffers_skip_updates_and_fallback_is_not_evidence():
    trainer, env, expert, *_ = fixture()
    planner = ASPPlanner(on_failure="base_policy")
    for stream in trainer._streams:
        stream.factory = lambda obs, info: TeachingSession(planner, lambda obs, info: PlanningProblem(":- ."))
    steps = []
    trainer.learn(2, callback=steps.append)
    assert env.actions == expert.actions == [0]
    assert steps[0].retained and not steps[1].retained
    assert trainer.stats.filtered_steps == 0
    assert trainer.stats.ordinary_fallbacks == trainer.stats.expert_fallbacks == 1
    assert trainer.stats.expert_buffer_size == 0
    assert trainer.stats.updates == 2  # Ordinary TD remains valid task evidence.
    trainer, *_ = fixture()
    trainer.learn(1)
    assert trainer.stats.updates == trainer.stats.expert_buffer_size == trainer.stats.ordinary_buffer_size == 0


def test_mutable_observations_detached_before_step_and_ring_capacity():
    trainer, *_ = fixture(filter_replay=False, buffer_size=2, learning_rate=0)
    trainer.learn(8)
    assert trainer.stats.ordinary_buffer_size == trainer.stats.expert_buffer_size == 2
    for buffer in trainer._buffers:
        assert {t.state for t in buffer} == {_state_key(np.array([2])), _state_key(np.array([3]))}
        assert {t.next_state for t in buffer} == {_state_key(np.array([3])), _state_key(np.array([4]))}


def test_resume_budget_reset_history_and_no_reseeding():
    trainer, env, expert, sessions, inputs = fixture(env=TinyEnv(truncated=True), expert=TinyEnv(truncated=True))
    trainer.learn(3)
    before = trainer.stats
    trainer.learn(3)
    assert trainer.stats == before and len(sessions) == 3
    trainer.learn(5)
    assert trainer.stats.total_steps == 5
    assert trainer.stats.ordinary_steps == 3 and trainer.stats.expert_steps == 2
    assert len(sessions) == 5 and all(len(history) == 1 for history in inputs)
    assert all(history[0].truncated for history in inputs)
    assert env.seeds[1:] == [None, None] and expert.seeds[1:] == [None]


def test_training_rng_independent_from_prediction_and_reproducible():
    a, env_a, expert_a, *_ = fixture(epsilon=0.5, expert_epsilon=0.5)
    b, env_b, expert_b, *_ = fixture(epsilon=0.5, expert_epsilon=0.5)
    a.learn(12)
    b.learn(12, callback=lambda step: [b.policy.predict(np.array([0]), deterministic=False) for _ in range(5)] and None)
    assert env_a.actions == env_b.actions and expert_a.actions == expert_b.actions
    assert env_a.draws == env_b.draws and expert_a.draws == expert_b.draws
    assert env_a.draws != expert_a.draws
    for state, row in a.policy.q_table.items():
        np.testing.assert_array_equal(row, b.policy.q_table[state])


def test_offset_actions_and_current_masks():
    trainer, env, expert, *_ = fixture(env=TinyEnv(start=7), expert=TinyEnv(start=7))
    trainer.learn(2)
    assert env.actions == [7] and expert.actions == [8]
    trainer, env, expert, *_ = fixture()
    trainer.policy.use_action_mask = True
    original = env.reset
    env.reset = lambda **kw: (original(**kw)[0], {"action_mask": [0, 1]})
    trainer.learn(1)
    assert env.actions == [1] and trainer.stats.filtered_steps == 0


def test_next_action_mask_controls_truncated_bootstrap():
    trainer, _env, expert, *_ = fixture(expert=TinyEnv(truncated=True), expert_weight=0)
    trainer.policy.use_action_mask = True
    original = expert.step

    def step(action):
        obs, reward, terminated, truncated, info = original(action)
        info["action_mask"] = [0, 1]
        return obs, reward, terminated, truncated, info

    expert.step = step
    trainer._target[_state_key(np.array([1]))] = np.array([100, 4])
    trainer.learn(2)
    assert trainer.policy.q_values(np.array([0]))[1] == pytest.approx(0.3)


def test_callback_stops_after_completed_step_and_validates_return():
    trainer, *_ = fixture()
    trainer.learn(10, callback=lambda step: False)
    assert trainer.stats.total_steps == 1
    with pytest.raises(TypeError, match="callback must return"):
        trainer.learn(10, callback=lambda step: 42)
    assert trainer.stats.total_steps == 2


@pytest.mark.parametrize(
    "option,value",
    [
        ("buffer_size", 0),
        ("batch_size", -1),
        ("target_update_interval", 0),
        ("epsilon", -0.1),
        ("expert_epsilon", 1.1),
        ("margin", -1),
        ("expert_weight", -1),
        ("filter_replay", 1),
        ("seed", True),
        ("margin", float("nan")),
        ("batch_size", True),
    ],
)
def test_configuration_validation(option, value):
    with pytest.raises((ValueError, TypeError)):
        fixture(**{option: value})


def test_environment_and_factory_validation():
    env = TinyEnv()
    with pytest.raises(ValueError, match="independent"):
        fixture(env=env, expert=gym.Wrapper(env))
    with pytest.raises(ValueError, match="action spaces"):
        fixture(expert=TinyEnv(start=1))
    trainer, *_ = fixture()
    trainer._streams[0].factory = lambda obs, info: None
    with pytest.raises(TypeError, match="factory must return"):
        trainer.learn(1)


def test_independent_sessions_required():
    trainer, *_ = fixture()
    session = TeachingSession(ASPPlanner(), lambda obs, info: PlanningProblem("candidate(0,0..1)."))
    for stream in trainer._streams:
        stream.factory = lambda obs, info: session
    with pytest.raises(ValueError, match="independent TeachingSession"):
        trainer.learn(2)


def test_saved_policy_inference_without_optional_dependencies(tmp_path):
    trainer, *_ = fixture()
    trainer.learn(4)
    path = tmp_path / "taught.zip"
    trainer.policy.save(path)
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'clingo', 'torch', 'stable_baselines3'}:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
import gymnasium as gym
import numpy as np
from npc_gym.algorithms import TabularQLearning, TabularOFTEN
class Env(gym.Env):
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Box(0, 100, (1,), dtype=np.int64)
policy = TabularQLearning.load(sys.argv[1], env=Env())
assert policy.predict(np.array([0])) == int(sys.argv[2])
assert np.allclose(policy.q_values(np.array([0])), [float(v) for v in sys.argv[3:]])
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(path),
            str(trainer.policy.predict(np.array([0]))),
            *(str(value) for value in trainer.policy.q_values(np.array([0]))),
        ],
        check=True,
    )
    # Loaded policy starts a new teaching run with explicit fresh environments.
    loaded = TabularQLearning.load(path, env=TinyEnv())
    fresh = TabularOFTEN(
        loaded,
        TinyEnv(),
        ordinary_session=trainer._streams[0].factory,
        expert_session=trainer._streams[1].factory,
        seed=52,
    )
    assert fresh.stats.total_steps == 0
    fresh.learn(2)
    assert fresh.stats.total_steps == 2


def test_combined_mean_gradient_uses_pre_update_values():
    trainer, *_ = fixture(filter_replay=False, batch_size=7, initial=(2, 0))
    steps = []
    trainer.learn(2, callback=steps.append)
    # First ordinary TD: 2 - .1*(2-1) = 1.9. Next update sums the
    # ordinary TD gradient .9 and expert gradient +1 on action 0.
    np.testing.assert_allclose(trainer.policy.q_values(np.array([0])), [1.71, 0.1])
    assert steps[1].ordinary_td_loss == pytest.approx(0.405)
    assert steps[1].expert_td_loss == pytest.approx(0.5)
    assert steps[1].expert_loss == pytest.approx(2.7)


def test_history_tracks_executed_proposal_instead_of_hypothetical_fixed_action():
    from npc_gym.envs import StormTaxiEnv
    from npc_gym.policy_fixes import TaxiModel

    ordinary, expert = StormTaxiEnv(), StormTaxiEnv()
    histories = []

    def factory(env):
        def start(obs, info):
            model = TaxiModel()
            histories.append(model)
            return TeachingSession(ASPPlanner(), lambda obs, info: model.problem(), model.advance)

        return start

    try:
        policy = TabularQLearning(ordinary, seed=12, log_interval=None)
        trainer = TabularOFTEN(
            policy, expert, ordinary_session=factory(ordinary), expert_session=factory(expert), seed=7
        )
        trainer.learn(6)
        assert len(histories) == 2 and histories[0] is not histories[1]
        assert trainer.stats.total_steps == 6 and trainer.stats.updates > 0
        assert policy.explored_states > 0
    finally:
        ordinary.close()
        expert.close()


def test_save_path_separation_overwrite_and_cleanup(tmp_path):
    trainer, *_ = fixture()
    base, taught = tmp_path / "base.zip", tmp_path / "taught.zip"
    trainer.policy.save(base)
    original = base.read_bytes()
    trainer.learn(2)
    trainer.policy.save(taught)
    previous = TabularQLearning.load(taught, env=TinyEnv()).q_values(np.array([0]))
    trainer.learn(4)
    trainer.policy.save(taught)
    assert base.read_bytes() == original
    assert set(tmp_path.iterdir()) == {base, taught}
    assert not np.array_equal(previous, TabularQLearning.load(taught, env=TinyEnv()).q_values(np.array([0])))
