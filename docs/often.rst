OFTEN teaching
==============

OFTEN (*on-the-fly teaching of ethical norms*) fine-tunes a task policy using
`policy fixes <policy_fixes.rst>`__. An ordinary environment executes the learner's
proposals. An independent demonstration environment collects norm-guided experience.
The learner learns task values from both streams and receives an additional
expert-action loss. Evaluation can then use the learned policy alone.

``DQNOFTEN`` adapts an SB3 DQN; ``TabularOFTEN`` is a tabular extension.
From a checkout, install ``python -m pip install '.[asp]'`` to construct its ASP planners. Importing the trainer or running a
saved tabular policy requires neither Clingo nor SB3. DQN teaching uses
``python -m pip install '.[asp,sb3]'``; a saved DQN needs SB3 but no Clingo. Sources and licensing are
recorded in `THIRD_PARTY_NOTICES.md <../THIRD_PARTY_NOTICES.md#often-and-policy-fixing>`__.

For a paired evaluation of a pretrained feature DQN, OFTEN and ordinary continued
DQN, see the
`Pacman example <examples.rst#feature-based-pacman-often>`__.

Learner observations
--------------------

The fixer can read a richer public state than the learner observes. Teaching
does not add that information to the learned policy. If two states have the same
learner observation but require different avoidance actions, the supplied
observation-based policies cannot distinguish them at inference. Teaching may
still improve average behavior, but it cannot reproduce every such fix.

For Gardener, use ``StateFeatureObsWrapper(env, include_frogs=True)``
from ``npc_gym.wrappers.gardener_wrappers`` to keep the compact task features
and add possible collection/drainage counts per action plus action legality.
These summaries expose the default one-step fixer's immediate risks. Use the same
encoding for the base, taught and comparison policies. Both Gardener observation
wrappers omit frogs by default. The `Gardener catalogue
<catalogue/gardener.rst#catalogue-gardener-policy-fixes>`__ describes the encoding,
its remaining observation limits and the three supported non-temporal norms.

Tabular training contract
-------------------------

``TabularOFTEN(policy, expert_env, *, ordinary_session, expert_session, ...)``
teaches an existing ``TabularQLearning`` instance **in place**. Its ``policy.env``
is the ordinary environment. Supply a second, independent Gymnasium environment
with matching observation and Discrete action spaces. Both environments must
implement the same task and observation encoding. Two wrappers around the same
underlying environment are rejected. Automatic-reset and vector environments are
unsupported; use ordinary environments, optionally with ``TimeLimit``.

Each session factory receives ``(reset_observation, reset_info)`` and returns a
fresh ``TeachingSession(planner, problem, advance=None)``. Its
``problem(observation, info)`` builds the current planning problem. If provided,
``advance(MonitorInput(...))`` consumes exactly one actual step, including its
termination/truncation flags. This requires ``info['labels']``. Factories initialize
domain history from the reset snapshot; reset labels are not passed to ``advance``.
Never share mutable history between streams, reuse it after reset, or step the
environment from a callback. See each domain's catalogue for initialization rules.

``learn(total_timesteps, callback=None)`` takes a **cumulative combined teaching
budget**, alternating ordinary and demonstration transitions, ordinary first.
An odd budget gives the ordinary stream one extra transition. Every nonempty call
starts fresh episodes; replay, frozen target values, counters and RNG streams
continue. The first reset of each stream uses a distinct seed derived from the
trainer's ``seed``. Subsequent resets use ``seed=None``. Starting teaching therefore
starts new environment streams even if the base policy was previously trained.
Exploration, replay sampling and ordinary inference have independent generators.
The caller owns and closes the environments.

* The ordinary stream executes its epsilon-greedy **proposal**. With replay filtering
  enabled, a proven-optimal fixer decision that differs from that proposal discards
  the transition from ordinary replay. Observed violations do not decide retention.
  Filtering does not undo the action or remove its reward from diagnostics.
* The demonstration stream executes the fixer's action. All certified decisions,
  including unchanged proposals, enter expert replay. Explicit solver fallbacks
  execute the proposal but never enter expert replay. An ordinary fallback remains
  ordinary task evidence and never causes filtering.
* Task rewards are used unchanged. Both streams use the current learner's values;
  teaching does not keep a separate frozen base policy for proposals.
* When ``policy.use_action_mask`` is enabled, proposals and the current fixed action
  obey the current mask; nonterminal TD bootstrapping obeys the next mask. Future
  planning admissibility belongs to the domain model. The expert loss considers
  all actions, following the upstream loss, even when masks restrict execution.

A small complete example
-------------------------

This runs Warn planning and tabular updates on two Taxi environments,
then saves and reloads the taught policy. Replace the base learner's short training
budget for a substantive experiment. This illustrates the teaching lifecycle;
learning the exact warning policy also requires observations that distinguish
rain onset from persistent rain (see `Taxi policy fixes <catalogue/taxi.rst#taxi-policy-fixes-often>`__).

.. code-block:: python

   from pathlib import Path
   from tempfile import TemporaryDirectory

   from gymnasium.wrappers import TimeLimit

   from npc_gym.algorithms import TabularOFTEN, TabularQLearning, TeachingSession
   from npc_gym.envs import StormTaxiEnv
   from npc_gym.policy_fixes import ASPPlanner, TaxiModel

   ordinary = StormTaxiEnv()
   expert = StormTaxiEnv()
   ordinary_env = TimeLimit(ordinary, max_episode_steps=8)
   expert_env = TimeLimit(expert, max_episode_steps=8)

   def session(observation, info):
       model = TaxiModel()
       return TeachingSession(ASPPlanner(), lambda obs, info: model.problem(), model.advance)

   try:
       policy = TabularQLearning(ordinary_env, seed=7, log_interval=None).learn(8)
       teacher = TabularOFTEN(
           policy, expert_env,
           ordinary_session=session,
           expert_session=session,
           batch_size=2, target_update_interval=4, seed=17,
       ).learn(16)
       assert teacher.stats.ordinary_steps == teacher.stats.expert_steps == 8
       assert teacher.stats.updates > 0
       with TemporaryDirectory() as directory:
           path = Path(directory) / 'taught.zip'
           policy.save(path)
           restored = TabularQLearning.load(path, env=ordinary_env)
           observation, info = ordinary_env.reset(seed=99)
           assert restored(observation, info) == policy(observation, info)
   finally:
       ordinary_env.close()
       expert_env.close()

Replay and tabular updates
--------------------------

Each stream has a separate bounded replay buffer, replacing the oldest transition
when full. After every environment transition, sample ``batch_size`` items **with
replacement from each nonempty buffer** and perform one combined update. An empty
buffer contributes zero loss; if both are empty, skip the update and target-sync
counter. Discarded ordinary steps can still trigger learning from existing replay.
Structured observations are encoded at collection time, so later environment
mutation cannot change stored states.

Let Q denote the online table, Q-minus the periodically frozen target table, and
``aE`` the recorded expert action. Each sample contributes:

.. code-block:: text

   y = reward + gamma * max_allowed Q-minus(next_state, action)
   TD loss = 0.5 * (Q(state, executed_action) - y)^2
   expert loss = abs(max_action(Q(state, action) + margin(action, aE))
                     - Q-minus(state, aE))

For a terminated transition, ``y = reward``. Truncation alone bootstraps from the
actual final observation; if both flags are true, termination takes precedence.
The margin is zero for ``aE`` and ``margin`` for every other action. Unseen target
rows are zero. The frozen target initially copies the supplied policy.

The total loss is the sum of the ordinary mean TD loss, expert mean TD loss, and
``expert_weight`` times the mean expert loss. All gradients use the same pre-update
online values. Multiply the summed gradient by ``policy.learning_rate`` and subtract
it; ``policy.gamma`` sets the discount. There is no gradient through the target.
The absolute-value subgradient is -1, 0, or +1 according to its signed argument.
Only the maximizing online action receives the expert subgradient; ties choose
the lowest action ID. In particular, this upstream loss can lower a competing
value without directly raising the expert value. It is not the usual hinge loss
against the online expert value. Greedy proposal ties are instead sampled with
the stream's seeded RNG, matching tabular exploration.

After every ``target_update_interval`` successful replay updates, copy the entire
online table to the target. This tabular extension uses one-step TD and the stated
subgradient; it does not claim equivalence to DQN's optimizer or learning curves.

Tabular options and diagnostics
-------------------------------

.. list-table:: Trainer options
   :header-rows: 1

   * - Option
     - Default
     - Meaning
   * - ``buffer_size``
     - 10,000
     - Capacity of each stream's buffer, in transitions.
   * - ``batch_size``
     - 32
     - Samples per nonempty buffer per update.
   * - ``target_update_interval``
     - 100
     - Replay updates between frozen-target copies.
   * - ``epsilon``, ``expert_epsilon``
     - 0.1, 0.0
     - Constant proposal exploration probabilities in the two streams.
   * - ``margin``, ``expert_weight``
     - 0.8, 1.0
     - Nonnegative expert margin and loss coefficient.
   * - ``filter_replay``
     - ``True``
     - Discard ordinary replay on certified fixer disagreement.
   * - ``seed``
     - ``None``
     - Seed for independent exploration, environment and replay generators.

For the no-expert-loss ablation, set ``expert_weight=0``; demonstration TD remains.
For no filtering, set ``filter_replay=False``; both streams still solve and report
fixer decisions. To train from scratch, supply an untrained ``TabularQLearning``.
Base-policy exploration schedules and its learning counters are not advanced by
this trainer. Do not simultaneously call ``policy.learn`` or mutate its table.

``teacher.stats`` is a detached ``TeachingStats`` snapshot. It reports actual
ordinary/expert steps and their sum, completed episodes, cumulative unmodified
returns (including partial episodes), certified disagreements, fallbacks, filtered
ordinary steps, current buffer sizes and replay updates. ``policy.num_timesteps``
continues to describe base training; report it separately from teaching budgets.

The optional callback receives ``TeachingStep`` after each collected transition
and its replay update. It contains the actual action, reward and ending flags,
retention decision, complete ``PlanningDecision``, unweighted mean losses and
current statistics. Return ``False`` to stop after that transition. Planned costs
and disagreement counts are not observed norm violations: use independent monitors
and `evaluation <evaluation.rst>`__ for compliance and task-return comparisons.
A changed ordinary decision records a proposed intervention, although that stream
executes the original proposal.

Saving and further teaching
---------------------------

Save ``teacher.policy`` with ``TabularQLearning.save`` and reload with
``TabularQLearning.load(..., env=...)``. This is the existing safe ZIP/JSON/NumPy
policy format. It contains learned values, not environments, callbacks, Clingo
objects, replay, teaching counters or target tables. It uses the existing overwrite
behavior and creates no separate teaching artifacts. Ordinary inference needs no
ASP dependency. For further teaching, construct a new ``TabularOFTEN`` with the
loaded policy, fresh environments and session factories. That starts a new teaching
run with empty replay and a fresh target copy, not an exact continuation checkpoint.


DQN teaching
------------

``DQNOFTEN(model, ordinary_env, expert_env, *, ordinary_session, expert_session,
margin=50, initial_expert_weight=0.1, final_expert_weight=1, filter_replay=True,
use_action_mask=False, seed=None)`` teaches an SB3 ``DQN`` **in place**. Import it
from ``npc_gym.integrations.sb3``. The online network, target network and optimizer
state are retained. Two fresh replay buffers replace the training data for
teaching; the model's base replay and counters remain unchanged.

Configure ordinary DQN settings on ``model``. The trainer uses its replay
capacity, batch size, warm-up, learning-rate and exploration schedules, discount,
rollout frequency, gradient-step count, target interval, ``tau`` and gradient
clipping. A model constructed with SB3 defaults therefore keeps those defaults.
Loading a checkpoint retains its saved settings, including any custom settings.
Configure the model before constructing its trainer; do not train or mutate it
concurrently.

Supply two independent ordinary Gymnasium environments with matching observations
and zero-based Discrete actions, using the session factories described above.
Vectorized or automatic-reset collection, multi-environment models,
``VecNormalize`` and ``n_steps>1`` are unsupported. Box, Discrete, MultiDiscrete and
dictionary observations are supported, including ``MultiInputPolicy``. The caller
owns and closes both environments.

Collection and learning
~~~~~~~~~~~~~~~~~~~~~~~

The trainer collects one ordinary rollout followed by one expert rollout, each
using ``model.train_freq``. With SB3 defaults this is four ordinary transitions,
four expert transitions, then one gradient update. ``gradient_steps=-1`` instead
performs one update per transition in the completed expert rollout. Episode-based
rollout frequencies are also supported. Learning starts only after the ordinary
transition count exceeds ``model.learning_starts`` and both replay buffers contain
eligible experience.

* During warm-up, both streams execute random actions. Afterwards, both use the
  model's epsilon-greedy exploration schedule. A greedy ordinary action is executed
  unchanged; a greedy expert action is replaced by the planner's action. Random
  expert actions are executed directly and checked for replay eligibility.
* With filtering enabled, ordinary actions and random expert actions enter replay
  only when the certified fixer agrees with the executed action. Greedy certified
  fixes enter expert replay. Turning filtering off retains disagreements too.
  Explicit planner fallbacks remain ordinary evidence but never expert evidence.
* Each buffer's capacity counts **all collected transitions**, including rejected
  ones. Rejected transitions evict old evidence but cannot be sampled. Sampling is
  uniform with replacement among eligible entries, with equal batch sizes from
  the two buffers. If either has no eligible entries, the update is skipped.
* Each update sums ordinary Huber TD loss, expert Huber TD loss and the weighted
  expert loss. The detached-target absolute expert formula is the one shown in
  the tabular section. It follows the released implementation, whose formula
  differs from the paper's online-value hinge formulation. No n-step loss is added.
  Task rewards remain unchanged.

The expert coefficient changes linearly from ``initial_expert_weight`` to
``final_expert_weight``. The paper uses 0.1 to 1 for Pacman and 0.1 to 0.5 for
Gardener and SUMO. Its margins are respectively 50, 5 and 10 in unscaled reward
units. Pacman's example divides training rewards by 100 and uses ``margin=0.5``;
evaluation reports raw task rewards.
Scaling rewards and margin together does not make training equivalent across
reward units: Huber TD and expert losses scale differently. Set both expert weights
to zero for the no-expert-loss ablation; both TD terms remain.

Target updates occur every ``model.target_update_interval`` **combined
transitions**, whether or not replay updates run. They use ``model.tau`` and copy
running normalization statistics as SB3 does. Learning-rate and exploration
schedules restart for teaching. As in the reference, schedule progress counts
ordinary transitions, temporarily including the current expert rollout while it
is collected. The expert coefficient uses the same progress. The model's original
training counters do not determine teaching progress.

Budgets, evaluation and saving
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``learn(total_timesteps, *, schedule_timesteps=None, callback=None)`` uses a
**cumulative combined transition budget**. A balanced run of two million
transitions corresponds to the reference's one million ordinary plus one million
expert transitions. An exact budget may stop partway through a rollout. Episodes,
partial rollouts, replay, target values and RNGs continue across calls.

``schedule_timesteps`` sets the full combined horizon for the schedules; it
defaults to ``total_timesteps`` and cannot be smaller. For a short diagnostic run,
use ``teacher.learn(100_000, schedule_timesteps=2_000_000)``. Further calls with the
same schedule horizon continue that run without accelerating its schedules.
Evaluate on a separate environment to preserve the collection state.

``teacher.stats`` and the callback use ``TeachingStats`` and ``TeachingStep`` as
above. The callback runs after every collected transition and any resulting
training phase; its losses describe the last gradient update in that phase, or
zero when none ran. Return ``False`` to stop. ``teacher.exploration_rate`` and
``teacher.expert_weight`` expose the current schedule values.
``q_values(observation)`` returns detached values for one unbatched observation.
Inference restores network modes; training isolates its NumPy and Torch RNGs.

With ``use_action_mask=True``, proposals, fixed actions and nonterminal TD targets
respect the supplied masks, including at truncation. The expert loss considers
all actions, matching the reference loss. Standard SB3 inference itself ignores
these masks. CPU and CUDA are supported; automated checks use CPU.

Save ``teacher.model`` using SB3's ``save`` and reload with ``DQN.load``. The saved
policy runs without a planner or Clingo. The file retains online/target weights
and optimizer state, but no teaching replay, environments, counters or schedules.
A new trainer therefore starts a new teaching run, not an exact resumption.

Using the Taxi session factory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This continues the ``sessions`` definition above, using **fresh environments**
and standard DQN settings. It trains a small example from scratch; a supplied
pretrained DQN is taught in the same way.

.. code-block:: python

   from stable_baselines3 import DQN
   from npc_gym.integrations.sb3 import DQNOFTEN

   ordinary = StormTaxiEnv()
   expert = StormTaxiEnv()
   ordinary_env = TimeLimit(ordinary, max_episode_steps=8)
   expert_env = TimeLimit(expert, max_episode_steps=8)
   try:
       model = DQN('MlpPolicy', ordinary_env, device='cpu', seed=7)
       teacher = DQNOFTEN(
           model, ordinary_env, expert_env,
           ordinary_session=session,
           expert_session=session, margin=1, seed=17,
       ).learn(256)
       assert teacher.stats.total_steps == 256 and teacher.stats.updates > 0
       with TemporaryDirectory() as directory:
           path = Path(directory) / 'taught-dqn.zip'
           model.save(path)
           restored = DQN.load(path, device='cpu')
           observation, info = ordinary_env.reset(seed=99)
           assert restored.predict(observation, deterministic=True)[0] == model.predict(
               observation, deterministic=True
           )[0]
   finally:
       ordinary_env.close()
       expert_env.close()

Reference fidelity
~~~~~~~~~~~~~~~~~~

The integration follows the reference's loss, expert exploration, filtering,
paired rollout/update schedule and gradual expert weighting. It uses independent
environments and uniform replay as intended by the method. In the released
revision, the expert environment aliases the ordinary environment, and the replay
sampler's nested loops repeatedly select one time index within a batch. Those two
implementation defects are not reproduced. Empty eligible buffers safely skip
updates instead of entering an unbounded sampling loop. Sources and the inspected
revision are recorded in
`the third-party notices <../THIRD_PARTY_NOTICES.md#often-and-policy-fixing>`__.

These algorithm checks do not establish learning-curve equivalence. Training
results also depend on observations, reward scaling, DQN configuration and seeds.
