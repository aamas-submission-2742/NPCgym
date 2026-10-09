"""CPU numerical and collection tests for the optional DQN OFTEN trainer."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

if os.environ.get("NPC_GYM_REQUIRE_OFTEN") == "1":
    import clingo  # noqa: F401 -- fail the integration gate if extras are missing
    import stable_baselines3
    import torch
else:
    pytest.importorskip("clingo")
    stable_baselines3 = pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.dqn.policies import DQNPolicy

from npc_gym.algorithms import TeachingSession
from npc_gym.integrations.sb3 import DQNOFTEN
from npc_gym.integrations.sb3.often import _expert_loss, _Transition
from npc_gym.policy_fixes import ASPPlanner, PlanningProblem


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyEnv(gym.Env):
    def __init__(self, *, terminated=False, truncated=False, reward=1.0, kind="box"):
        self.action_space = gym.spaces.Discrete(2)
        self.kind = kind
        box = gym.spaces.Box(0, 100, (1,), dtype=np.float32)
        self.observation_space = {
            "box": box,
            "dict": gym.spaces.Dict({"position": box}),
            "discrete": gym.spaces.Discrete(100),
            "multi": gym.spaces.MultiDiscrete([100]),
        }[kind]
        self.terminated, self.truncated, self.reward = terminated, truncated, reward
        self.actions, self.seeds, self.draws = [], [], []
        self.observation = np.zeros(1, dtype=np.float32)
        self.mask = [1, 1]

    def _obs(self):
        if self.kind == "dict":
            return {"position": self.observation}
        if self.kind == "discrete":
            return int(self.observation[0])
        if self.kind == "multi":
            return self.observation.astype(np.int64)
        return self.observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.seeds.append(seed)
        self.observation[:] = 0
        return self._obs(), {"labels": frozenset(), "action_mask": self.mask}

    def step(self, action):
        self.actions.append(action)
        self.draws.append(self.np_random.random())
        self.observation += 1
        return (
            self._obs(),
            self.reward,
            self.terminated,
            self.truncated,
            {
                "labels": frozenset({action}),
                "action_mask": self.mask,
            },
        )


def make_model(env, *, initial=(2.0, 0.0), **options):
    config = {
        "learning_rate": 0.1,
        "gamma": 0.5,
        "buffer_size": 8,
        "batch_size": 1,
        "learning_starts": 0,
        "train_freq": 1,
        "exploration_initial_eps": 0,
        "exploration_final_eps": 0,
    }
    config.update(options)
    model = stable_baselines3.DQN(
        "MultiInputPolicy" if isinstance(env.observation_space, gym.spaces.Dict) else "MlpPolicy",
        env,
        device="cpu",
        seed=17,
        policy_kwargs={"net_arch": [], "optimizer_class": torch.optim.SGD},
        **config,
    )
    with torch.no_grad():
        for param in model.q_net.parameters():
            param.zero_()
        model.q_net.q_net[-1].bias.copy_(torch.tensor(initial))
    model.q_net_target.load_state_dict(model.q_net.state_dict())
    return model


def fixture(*, ordinary=None, expert=None, initial=(2.0, 0.0), **kwargs):
    ordinary = TinyEnv() if ordinary is None else ordinary
    expert = TinyEnv() if expert is None else expert
    model_options = kwargs.pop("model_options", {})
    model = make_model(ordinary, initial=initial, **model_options)
    sessions, histories = [], []
    problem = PlanningProblem('candidate(0,0..1). cost("violations",1,harm) :- action(0,0).')

    def factory(obs, info):
        history = []
        session = TeachingSession(ASPPlanner(), lambda obs, info: problem, history.append)
        histories.append(history)
        sessions.append(session)
        return session

    config = {"margin": 0.8, "initial_expert_weight": 1, "final_expert_weight": 1, "filter_replay": False, "seed": 41}
    config.update(kwargs)
    trainer = DQNOFTEN(model, ordinary, expert, ordinary_session=factory, expert_session=factory, **config)
    return trainer, ordinary, expert, sessions, histories


def test_two_stream_actions_filtering_history_budgets_and_numerical_update():
    trainer, ordinary, expert, sessions, histories = fixture()
    steps = []
    trainer.learn(2, callback=steps.append)
    assert ordinary.actions == [0] and expert.actions == [1]
    assert steps[0].retained and steps[1].retained
    assert sessions[0] is not sessions[1]
    assert histories[0][0].labels == frozenset({0})
    assert histories[1][0].labels == frozenset({1})
    assert trainer.stats.ordinary_steps == trainer.stats.expert_steps == 1
    assert trainer.stats.total_steps == 2 and trainer.stats.updates == 1
    assert trainer.stats.ordinary_return == trainer.stats.expert_return == 1
    assert trainer.stats.ordinary_interventions == trainer.stats.expert_interventions == 1
    assert ordinary.seeds[0] != expert.seeds[0]
    # y=1+.5*2=2, Huber gradient -1; margin gradient +1 on action 0.
    np.testing.assert_allclose(trainer.q_values(np.array([0], dtype=np.float32)), [1.9, 0.1])
    assert steps[1].expert_td_loss == pytest.approx(1.5)
    assert steps[1].expert_loss == pytest.approx(2.8)
    assert trainer.model.num_timesteps == 0
    assert trainer.model.replay_buffer.size() == 0
    assert all(param.grad is None for param in trainer.model.q_net_target.parameters())


@pytest.mark.parametrize("target,expected", [([0, 0], [1, 0]), ([0, 4], [-1, 0]), ([0, 3], [0, 0])])
def test_exact_absolute_expert_gradient_and_detachment(target, expected):
    online = torch.tensor([[2.0, 0.0]], requires_grad=True)
    frozen = torch.tensor([target], dtype=torch.float32, requires_grad=True)
    loss = _expert_loss(online, frozen, torch.tensor([[1]]), 1)
    loss.backward()
    torch.testing.assert_close(online.grad, torch.tensor([expected], dtype=torch.float32))
    assert frozen.grad is None


def test_expert_max_ties_choose_lowest_action_and_batch_mean():
    online = torch.tensor([[0.0, 1.0], [2.0, 0.0]], requires_grad=True)
    loss = _expert_loss(online, torch.zeros_like(online), torch.tensor([[1], [0]]), 1)
    assert loss.item() == 1.5
    loss.backward()
    torch.testing.assert_close(online.grad, torch.tensor([[0.5, 0], [0.5, 0]]))


@pytest.mark.parametrize("terminated,truncated,td", [(True, False, 0.125), (False, True, 0.28125), (True, True, 0.125)])
def test_terminal_and_truncated_td(terminated, truncated, td):
    trainer, *_ = fixture(
        ordinary=TinyEnv(terminated=terminated, truncated=truncated, reward=0.5),
        expert=TinyEnv(terminated=terminated, truncated=truncated, reward=0.5),
        initial=(0.5, 0),
        initial_expert_weight=0,
        final_expert_weight=0,
    )
    steps = []
    trainer.learn(2, callback=steps.append)
    assert steps[-1].expert_td_loss == pytest.approx(td)
    expected = 0.05 if terminated else 0.075
    assert trainer.q_values(np.array([0]))[1] == pytest.approx(expected)


def test_target_sync_uses_transitions_and_tau_even_without_updates():
    trainer, *_ = fixture(model_options={"target_update_interval": 2, "tau": 0.25, "gradient_steps": 0})
    with torch.no_grad():
        for parameter in trainer.model.q_net_target.parameters():
            parameter.zero_()
    trainer.learn(1)
    assert trainer.model.q_net_target.q_net[-1].bias[0] == 0
    trainer.learn(2)
    assert trainer.model.q_net_target.q_net[-1].bias[0].item() == pytest.approx(0.5)
    trainer.learn(4)
    assert trainer.model.q_net_target.q_net[-1].bias[0].item() == pytest.approx(0.875)
    assert trainer.stats.updates == 0


@pytest.mark.parametrize("filtering,weight", [(True, 0), (False, 0), (False, 1)])
def test_ablations(filtering, weight):
    trainer, ordinary, expert, *_ = fixture(
        filter_replay=filtering, initial_expert_weight=weight, final_expert_weight=weight
    )
    trainer.learn(2)
    assert ordinary.actions == [0] and expert.actions == [1]
    assert trainer.stats.ordinary_buffer_size == int(not filtering)
    assert trainer.stats.expert_buffer_size == 1
    if filtering:
        assert trainer.stats.updates == 0  # No ordinary evidence to pair with the expert batch.
        np.testing.assert_array_equal(trainer.q_values(np.array([0])), [2, 0])
    else:
        assert trainer.q_values(np.array([0]))[1] > 0


def test_empty_buffers_learning_starts_and_fallback():
    trainer, *_ = fixture(model_options={"learning_starts": 3})
    trainer.learn(6)
    assert trainer.stats.updates == 0
    trainer.learn(8)
    assert trainer.stats.updates == 1
    trainer, *_ = fixture(filter_replay=True)
    fallback = ASPPlanner(on_failure="base_policy")
    for stream in trainer._streams:
        stream.factory = lambda obs, info: TeachingSession(fallback, lambda obs, info: PlanningProblem(":- ."))
    trainer.learn(2)
    assert trainer.stats.expert_buffer_size == 0 and trainer.stats.filtered_steps == 0
    assert trainer.stats.ordinary_fallbacks == trainer.stats.expert_fallbacks == 1
    assert trainer.stats.ordinary_buffer_size == 1 and trainer.stats.updates == 0


def test_huber_mean_and_combined_preupdate_gradient():
    trainer, *_ = fixture(model_options={"batch_size": 7})
    steps = []
    trainer.learn(2, callback=steps.append)
    # ordinary y=2 so gradient zero on first step; expert margin then lowers 0.
    np.testing.assert_allclose(trainer.q_values(np.array([0])), [1.9, 0.1])
    assert steps[1].ordinary_td_loss == pytest.approx(0)
    assert steps[1].expert_td_loss == pytest.approx(1.5)


@pytest.mark.parametrize("kind", ["box", "dict", "discrete", "multi"])
def test_observation_preprocessing_replay_copies_and_capacity(kind):
    trainer, ordinary, expert, *_ = fixture(
        ordinary=TinyEnv(kind=kind),
        expert=TinyEnv(kind=kind),
        model_options={"buffer_size": 2},
    )
    trainer.learn(8)
    assert trainer.stats.ordinary_buffer_size == trainer.stats.expert_buffer_size == 2
    assert trainer.stats.updates > 0
    for buffer in trainer._buffers:
        states = [item.state["position"] if kind == "dict" else item.state for item in buffer.entries]
        np.testing.assert_array_equal(np.asarray(states).flatten(), [2, 3])
    assert ordinary.actions and expert.actions


def test_masks_restrict_proposals_fixes_and_bootstrap():
    ordinary, expert = TinyEnv(), TinyEnv(truncated=True)
    trainer, *_ = fixture(
        ordinary=ordinary, expert=expert, use_action_mask=True, initial_expert_weight=0, final_expert_weight=0
    )
    ordinary.mask = expert.mask = [0, 1]
    steps = []
    trainer.learn(2, callback=steps.append)
    assert ordinary.actions == expert.actions == [1]
    assert trainer.stats.filtered_steps == 0
    # Both batches see pre-update action1=0; masked bootstrapping gives y=1.
    assert steps[-1].expert_td_loss == pytest.approx(0.5)


def test_terminal_mask_can_be_empty():
    trainer, _ordinary, expert, *_ = fixture(expert=TinyEnv(terminated=True), use_action_mask=True)
    original = expert.step

    def step(action):
        obs, reward, term, trunc, info = original(action)
        info["action_mask"] = [0, 0]
        return obs, reward, term, trunc, info

    expert.step = step
    trainer.learn(2)
    assert trainer.stats.updates == 1


def test_seeded_training_ignores_process_numpy_torch_and_evaluation_draws():
    first, ordinary1, expert1, *_ = fixture(
        model_options={"exploration_initial_eps": 0.5, "exploration_final_eps": 0.5}
    )
    second, ordinary2, expert2, *_ = fixture(
        model_options={"exploration_initial_eps": 0.5, "exploration_final_eps": 0.5}
    )
    first.learn(12)

    def evaluate(step):
        np.random.random(7)
        torch.rand(7)
        second.model.predict(np.array([0], dtype=np.float32), deterministic=False)

    second.learn(12, callback=evaluate)
    assert ordinary1.actions == ordinary2.actions and expert1.actions == expert2.actions
    assert ordinary1.draws == ordinary2.draws and expert1.draws == expert2.draws
    assert ordinary1.draws != expert1.draws
    for name, param in first.model.q_net.state_dict().items():
        torch.testing.assert_close(param, second.model.q_net.state_dict()[name], rtol=0, atol=0)


def test_dropout_training_preserves_global_rng_and_network_modes():
    class DropoutPolicy(DQNPolicy):
        def make_q_net(self):
            net = super().make_q_net()
            net.q_net.insert(0, torch.nn.Dropout(0.5))
            return net

    ordinary, expert = TinyEnv(), TinyEnv()
    model = stable_baselines3.DQN(DropoutPolicy, ordinary, device="cpu", seed=7, buffer_size=8)
    problem = PlanningProblem("candidate(0,0..1).")
    factory = lambda obs, info: TeachingSession(ASPPlanner(), lambda obs, info: problem)
    trainer = DQNOFTEN(model, ordinary, expert, ordinary_session=factory, expert_session=factory, seed=7)
    model.policy.train()
    model.q_net_target.eval()
    modes = [module.training for module in model.policy.modules()]
    numpy_state = deepcopy(np.random.get_state())
    torch_state = torch.random.get_rng_state().clone()
    trainer.learn(4)
    assert [module.training for module in model.policy.modules()] == modes
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    first, second = trainer.q_values(np.array([1])), trainer.q_values(np.array([1]))
    np.testing.assert_array_equal(first, second)
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state)


def test_cumulative_budget_callbacks_reset_sessions_and_seeds():
    trainer, ordinary, expert, sessions, histories = fixture(
        ordinary=TinyEnv(truncated=True), expert=TinyEnv(terminated=True)
    )
    trainer.learn(10, callback=lambda step: False)
    assert trainer.stats.total_steps == 1
    trainer.learn(1)
    assert len(sessions) == 1
    trainer.learn(5)
    assert trainer.stats.ordinary_steps == 3 and trainer.stats.expert_steps == 2
    assert all(len(history) == 1 for history in histories)
    assert ordinary.seeds[1:] == [None, None] and expert.seeds[1:] == [None]
    with pytest.raises(TypeError, match="callback must return"):
        trainer.learn(10, callback=lambda step: 1)


@pytest.mark.parametrize(
    "key,value",
    [
        ("margin", -1),
        ("initial_expert_weight", float("nan")),
        ("final_expert_weight", -1),
        ("filter_replay", 1),
        ("use_action_mask", 1),
        ("seed", True),
    ],
)
def test_invalid_configuration(key, value):
    with pytest.raises((TypeError, ValueError)):
        fixture(**{key: value})


def test_rejects_shared_env_vectorization_autoreset_normalization_and_nstep():
    trainer, ordinary, expert, *_ = fixture()
    factories = {"ordinary_session": trainer._streams[0].factory, "expert_session": trainer._streams[1].factory}
    with pytest.raises(ValueError, match="independent"):
        DQNOFTEN(trainer.model, ordinary, gym.Wrapper(ordinary), **factories)
    with pytest.raises(TypeError, match="vector"):
        DQNOFTEN(trainer.model, DummyVecEnv([TinyEnv]), expert, **factories)
    with pytest.raises(TypeError, match="Autoreset"):
        DQNOFTEN(trainer.model, gym.wrappers.Autoreset(ordinary), expert, **factories)
    multiple = stable_baselines3.DQN("MlpPolicy", DummyVecEnv([TinyEnv, TinyEnv]), device="cpu", buffer_size=8)
    with pytest.raises(ValueError, match="one environment"):
        DQNOFTEN(multiple, ordinary, expert, **factories)
    normalized = stable_baselines3.DQN("MlpPolicy", VecNormalize(DummyVecEnv([TinyEnv])), device="cpu", buffer_size=8)
    with pytest.raises(ValueError, match="VecNormalize"):
        DQNOFTEN(normalized, ordinary, expert, **factories)
    trainer.model.n_steps = 3
    with pytest.raises(ValueError, match="one-step"):
        DQNOFTEN(trainer.model, ordinary, expert, **factories)


def test_initial_online_target_weights_and_optimizer_are_retained():
    trainer, ordinary, expert, *_ = fixture()
    trainer.model.policy.optimizer = torch.optim.Adam(trainer.model.q_net.parameters(), lr=0.1)
    trainer.model.learning_starts = 0
    trainer.model.batch_size = 2
    trainer.model.learn(4)
    weights = deepcopy(trainer.model.q_net.state_dict())
    target = deepcopy(trainer.model.q_net_target.state_dict())
    optimizer = deepcopy(trainer.model.policy.optimizer.state_dict())
    fresh = DQNOFTEN(
        trainer.model,
        ordinary,
        expert,
        ordinary_session=trainer._streams[0].factory,
        expert_session=trainer._streams[1].factory,
    )
    for name, param in fresh.model.q_net.state_dict().items():
        torch.testing.assert_close(param, weights[name])
        torch.testing.assert_close(fresh.model.q_net_target.state_dict()[name], target[name])
    assert optimizer["state"]  # Base training really populated optimizer moments.
    restored_state = fresh.model.policy.optimizer.state_dict()["state"]
    for key, values in optimizer["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(restored_state[key][name], value)
    assert fresh.model.policy.optimizer.param_groups[0]["lr"] == 0.1
    assert fresh.stats.total_steps == 0 and fresh.model.num_timesteps == 4


def test_saved_sb3_inference_without_clingo_and_fresh_teaching(tmp_path):
    trainer, *_ = fixture()
    base, taught = tmp_path / "base.zip", tmp_path / "taught.zip"
    trainer.model.save(base)
    original = base.read_bytes()
    trainer.learn(4)
    expected = trainer.q_values(np.array([0]))
    trainer.model.save(taught)
    trainer.model.save(taught)  # Existing SB3 overwrite behavior; base remains separate.
    assert base.read_bytes() == original
    assert set(tmp_path.iterdir()) == {base, taught}
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'clingo':
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
import numpy as np
import torch
from stable_baselines3 import DQN
from npc_gym.integrations.sb3 import SB3Policy
model = DQN.load(sys.argv[1], device='cpu')
observation = np.array([0], dtype=np.float32)
with torch.no_grad():
    values = model.q_net(model.policy.obs_to_tensor(observation)[0]).cpu().numpy()[0]
assert np.allclose(values, np.asarray(sys.argv[2:], dtype=float))
assert SB3Policy(model)(observation, {}) == int(values.argmax())
assert 'clingo' not in sys.modules
""",
            str(taught),
            *(str(value) for value in expected),
        ],
        check=True,
    )
    loaded = stable_baselines3.DQN.load(taught, device="cpu")
    fresh = DQNOFTEN(
        loaded,
        TinyEnv(),
        TinyEnv(),
        ordinary_session=trainer._streams[0].factory,
        expert_session=trainer._streams[1].factory,
        seed=52,
    )
    fresh.learn(2)
    assert fresh.stats.total_steps == 2


@pytest.mark.parametrize(
    ("domain", "behavior"),
    [("gardener", "random"), ("pacman", "random"), ("pacman", "deterministic"), ("pacman", "partly-deterministic")],
)
def test_real_domain_sessions_train_without_changing_observations(domain, behavior):
    from npc_gym.envs import GardenerEnv, PacmanEnv
    from npc_gym.policy_fixes import GardenerModel, PacmanModel
    from npc_gym.wrappers.gardener_wrappers import StateFeatureObsWrapper

    def make():
        if domain == "gardener":
            raw = GardenerEnv(size=5)
            env = StateFeatureObsWrapper(raw, include_frogs=True)
        else:
            raw = PacmanEnv(layout="small", features="complete", ghost_behavior=behavior)
            env = raw
        return raw, gym.wrappers.TimeLimit(env, max_episode_steps=3)

    raw_ordinary, ordinary = make()
    raw_expert, expert = make()

    def sessions(raw):
        def start(obs, info):
            if domain == "gardener":
                planner_model = GardenerModel(raw.labeling_state(), horizon=1)
                planner = ASPPlanner(objectives=planner_model.objectives)
                return TeachingSession(
                    planner,
                    lambda obs, info: planner_model.problem(raw.labeling_state()),
                )
            planner_model = PacmanModel(raw.labeling_state(), horizon=1)
            return TeachingSession(ASPPlanner(), lambda obs, info: planner_model.problem(raw.labeling_state()))

        return start

    try:
        model = stable_baselines3.DQN(
            "MlpPolicy",
            ordinary,
            device="cpu",
            seed=4,
            buffer_size=8,
            batch_size=2,
            learning_starts=0,
            train_freq=1,
            exploration_initial_eps=0,
            exploration_final_eps=0,
            policy_kwargs={"net_arch": [8]},
        )
        trainer = DQNOFTEN(
            model,
            ordinary,
            expert,
            ordinary_session=sessions(raw_ordinary),
            expert_session=sessions(raw_expert),
            # This smoke checks training/observations; filtering is covered above.
            # A new maze can make all three ordinary actions disagree with the
            # planner, leaving no ordinary samples in such a short run.
            filter_replay=False,
            seed=7,
        )
        before = deepcopy(model.q_net.state_dict())
        trainer.learn(6)
        assert trainer.stats.total_steps == 6 and trainer.stats.updates > 0
        assert any(not torch.equal(param, before[name]) for name, param in model.q_net.state_dict().items())
        obs, _ = ordinary.reset(seed=11)
        assert ordinary.observation_space.contains(obs)
    finally:
        ordinary.close()
        expert.close()


def test_q_values_are_detached_and_reject_batched_inputs():
    trainer, *_ = fixture()
    observation = np.array([0], dtype=np.float32)
    values = trainer.q_values(observation)
    values[:] = 100
    np.testing.assert_array_equal(trainer.q_values(observation), [2, 0])
    with pytest.raises(ValueError, match="unbatched"):
        trainer.q_values(np.array([[0], [1]], dtype=np.float32))


@pytest.mark.parametrize("mask", [[0, 0], [1], [2, 1], None])
def test_invalid_action_masks(mask):
    trainer, ordinary, *_ = fixture(use_action_mask=True)
    ordinary.mask = mask
    with pytest.raises(ValueError, match="action_mask"):
        trainer.learn(1)


def test_network_failure_restores_modes_and_torch_randomness():
    trainer, *_ = fixture(filter_replay=False)
    saved = torch.random.get_rng_state().clone()
    modes = [module.training for module in trainer.model.policy.modules()]
    original = trainer.model.q_net.forward

    def forward(obs):
        if trainer.model.q_net.training:
            torch.rand(3)
            raise RuntimeError("test training failure")
        return original(obs)

    trainer.model.q_net.forward = forward
    with pytest.raises(RuntimeError, match="test training failure"):
        trainer.learn(2)
    assert [module.training for module in trainer.model.policy.modules()] == modes
    torch.testing.assert_close(torch.random.get_rng_state(), saved)


def test_rollouts_and_schedules_follow_reference_clock_with_a_longer_horizon():
    trainer, ordinary, expert, *_ = fixture(
        initial_expert_weight=0.1,
        final_expert_weight=1,
        model_options={
            "train_freq": 2,
            "gradient_steps": 0,
            "exploration_initial_eps": 1,
            "exploration_final_eps": 0.05,
        },
    )
    steps = []
    trainer.learn(4, schedule_timesteps=200, callback=steps.append)
    assert [step.stream for step in steps] == ["ordinary", "ordinary", "expert", "expert"]
    assert len(ordinary.actions) == len(expert.actions) == 2
    # Upstream's ordinary counter temporarily includes the current expert rollout.
    assert trainer.expert_weight == pytest.approx(0.136)
    assert trainer.exploration_rate == pytest.approx(0.62)
    with pytest.raises(ValueError, match="cover"):
        trainer.learn(5, schedule_timesteps=4)


@pytest.mark.parametrize("gradient_steps,expected", [(0, 0), (1, 2), (3, 6), (-1, 8)])
def test_gradient_frequency_follows_paired_rollouts(gradient_steps, expected):
    trainer, *_ = fixture(model_options={"train_freq": 4, "gradient_steps": gradient_steps})
    trainer.learn(16)
    assert trainer.stats.ordinary_steps == trainer.stats.expert_steps == 8
    assert trainer.stats.updates == expected


def test_episode_rollouts_use_model_frequency():
    trainer, *_ = fixture(
        ordinary=TinyEnv(terminated=True),
        expert=TinyEnv(truncated=True),
        model_options={"train_freq": (2, "episode")},
    )
    trainer.learn(8)
    assert trainer.stats.updates == 2
    assert trainer.stats.ordinary_episodes == trainer.stats.expert_episodes == 4


def test_expert_random_actions_are_executed_and_filtered():
    trainer, ordinary, expert, *_ = fixture(filter_replay=True, model_options={"learning_starts": 100})
    steps = []
    trainer.learn(64, callback=steps.append)
    assert set(expert.actions) == {0, 1}
    assert trainer.stats.updates == 0
    for step in steps:
        assert step.retained == (step.action == 1)
    assert trainer.stats.ordinary_buffer_size == ordinary.actions[-8:].count(1)
    assert trainer.stats.expert_buffer_size == expert.actions[-8:].count(1)


def test_rejected_visits_evict_old_replay():
    trainer, *_ = fixture(initial=(0, 2), filter_replay=True, model_options={"buffer_size": 2, "gradient_steps": 0})
    trainer.learn(2)
    assert trainer.stats.ordinary_buffer_size == 1
    with torch.no_grad():
        trainer.model.q_net.q_net[-1].bias.copy_(torch.tensor([2.0, 0.0]))
    trainer.learn(6)
    assert trainer.stats.ordinary_buffer_size == 0
    assert trainer.stats.expert_buffer_size == 2


def test_replay_batches_sample_across_collection_times():
    trainer, *_ = fixture(model_options={"batch_size": 256, "gradient_steps": 0})
    trainer.learn(16)
    batches = []
    hook = trainer.model.q_net.register_forward_pre_hook(lambda module, args: batches.append(args[0].clone()))
    try:
        trainer._update()
    finally:
        hook.remove()
    # Regression for the reference sampler: its nested loops repeated one time
    # index throughout most or all of a batch, despite having older eligible data.
    assert len(batches) == 2
    for batch in batches:
        np.testing.assert_array_equal(torch.unique(batch).numpy(), np.arange(8))


def test_prefix_calls_preserve_episodes_rollouts_and_randomness():
    options = {"train_freq": 3, "exploration_initial_eps": 0.5, "exploration_final_eps": 0.05}
    whole, ordinary, expert, *_ = fixture(model_options=options)
    split, ordinary_split, expert_split, *_ = fixture(model_options=options)
    whole.learn(24, schedule_timesteps=200)
    split.learn(5, schedule_timesteps=200)
    split.learn(24, schedule_timesteps=200)
    assert ordinary.actions == ordinary_split.actions and expert.actions == expert_split.actions
    assert len(ordinary_split.seeds) == len(expert_split.seeds) == 1
    assert whole.stats == split.stats
    for name, value in whole.model.policy.state_dict().items():
        torch.testing.assert_close(value, split.model.policy.state_dict()[name], rtol=0, atol=0)


def test_learning_rate_uses_model_schedule():
    trainer, *_ = fixture(model_options={"train_freq": 2, "learning_rate": lambda progress: 0.1 * progress})
    trainer.learn(4, schedule_timesteps=40)
    assert trainer.model.policy.optimizer.param_groups[0]["lr"] == pytest.approx(0.08)


def test_sb3_defaults_are_used_without_duplicate_trainer_parameters():
    ordinary, expert = TinyEnv(), TinyEnv()
    model = stable_baselines3.DQN("MlpPolicy", ordinary, seed=7, device="cpu")
    factory = lambda obs, info: TeachingSession(ASPPlanner(), lambda obs, info: PlanningProblem("candidate(0,0..1)."))
    trainer = DQNOFTEN(model, ordinary, expert, ordinary_session=factory, expert_session=factory, seed=7)
    trainer.learn(200)
    assert trainer.stats.updates == 0
    trainer.learn(208)
    assert trainer.stats.updates == 1
    assert trainer.buffer_size == model.buffer_size == 1_000_000
    assert trainer.batch_size == model.batch_size == 32
    assert model.target_update_interval == 10_000
    assert trainer.margin == 50 and trainer.initial_expert_weight == 0.1 and trainer.final_expert_weight == 1


def test_combined_updates_match_unmodified_reference_trainer():
    """Offline oracle: three SGD updates generated by upstream RuleDQfD.train.

    Fixed batches isolate loss, detachment, averaging and clipping from sampling.
    They contain terminal and continuing ordinary/expert transitions. Target and
    online weights differ, so replacing the upstream loss with a hinge is caught.
    """
    oracle = json.loads(Path(__file__).with_name("fixtures").joinpath("often_update.json").read_text())
    trainer, *_ = fixture(
        margin=oracle["margin"],
        model_options={"batch_size": 2, "max_grad_norm": oracle["max_grad_norm"]},
    )
    for net, values in ((trainer.model.q_net, oracle["initial"]), (trainer.model.q_net_target, oracle["target"])):
        with torch.no_grad():
            net.q_net[0].weight.copy_(torch.tensor(values["weight"]))
            net.q_net[0].bias.copy_(torch.tensor(values["bias"]))
    for buffer, batch in zip(trainer._buffers, oracle["batches"]):
        for index in range(2):
            buffer.append(
                _Transition(
                    np.array(batch["state"][index], dtype=np.float32),
                    batch["action"][index][0],
                    batch["reward"][index][0],
                    np.array(batch["next_state"][index], dtype=np.float32),
                    bool(batch["terminated"][index][0]),
                    (0, 1),
                )
            )
    trainer._replay_rng = SimpleNamespace(integers=lambda n, size: np.arange(size))
    for expected in oracle["updates"]:
        trainer.expert_weight = expected["expert_weight"]
        ordinary, expert, margin = trainer._update()
        assert ordinary + expert + expected["expert_weight"] * margin == pytest.approx(expected["loss"], abs=1e-6)
        torch.testing.assert_close(trainer.model.q_net.q_net[0].weight, torch.tensor(expected["weight"]))
        torch.testing.assert_close(trainer.model.q_net.q_net[0].bias, torch.tensor(expected["bias"]))
