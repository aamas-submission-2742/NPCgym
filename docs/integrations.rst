Integrations
============

These adapters connect NPC Gym's evaluation contracts to optional training and
optimization libraries. Install the corresponding extra before constructing an
adapter. Importing ``npc_gym`` itself loads neither optional dependency.

Stable-Baselines3
-----------------

Install the ``sb3`` extra with ``python -m pip install '.[sb3]'`` from a checkout. ``SB3Policy(model, deterministic=True)`` adapts ``model.predict`` to the environment-neutral
evaluation policy contract. It ignores ``info``; Stable-Baselines3 therefore does not consume NPC Gym action masks.

``DQNOFTEN`` fine-tunes an existing DQN with independent ordinary and fixed-action
demonstration streams. See `OFTEN teaching <often.rst#dqn-teaching>`__ for its session
contract, loss, budgets and save/reload behavior. It uses both ``asp`` and ``sb3``.

Recording and independent evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SB3EvaluationCallback`` can collect completed episodes from a vector training environment and run independent
intermediate evaluations at frequencies measured in individual environment transitions. Fresh training starts with an
evaluation at timestep zero; resumed training starts at the next future threshold. Each vector step triggers at most one
evaluation and hook call. The ``IntermediateEvaluation.thresholds`` tuple records
all crossed thresholds. For example, if one vector step advances from transition 8
to transition 12 at frequency 5, the callback evaluates once at 12 and records
threshold 10.

Supply ``evaluator`` and ``evaluation_frequency`` together to enable independent
evaluation. In this example, ``model`` is your SB3 model, ``eval_env`` is a separate
evaluation environment, and ``monitor_factories`` maps names to fresh-monitor factories:

.. code-block:: python

   from npc_gym.evaluation import evaluate
   from npc_gym.integrations.sb3 import SB3EvaluationCallback, SB3Policy

   def evaluate_model(model):
       return evaluate(eval_env, SB3Policy(model), episodes=20, monitors=monitor_factories, seed=10_000)

   callback = SB3EvaluationCallback(
       monitors=monitor_factories,
       evaluator=evaluate_model,
       evaluation_frequency=10_000,
   )
   model.learn(total_timesteps=100_000, callback=callback)
   training_summary = callback.training_summary()

Choosing the training monitor source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To record monitors that already run in the training environments, construct each ordinary environment with
``MonitorWrapper(env, monitors=monitor_factories)`` **inside** ``DummyVecEnv`` or ``SubprocVecEnv``, then use
``SB3EvaluationCallback(monitor_source="wrapper")``. Omit the callback's ``monitors`` argument; selecting both sources
is an error. The callback reads each slot's initial/reset information and terminal snapshot, preserving initial events
and final count snapshots across auto-reset without counting either twice. Bolt diagnostics are not included.
Source selection for a separate evaluator remains that evaluator's responsibility.

Resuming training and interpreting results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One callback records one ``learn`` call. Training-episode collection must start at timestep zero; for resumed learning,
use a fresh callback with ``collect_training_episodes=False``. Intermediate evaluation preserves Python, NumPy, CPU
Torch, action-space, and policy-mode state for seeded CPU PPO and DQN runs, but arbitrary evaluator mutations to the
model are outside the contract.

Training summaries total the rewards delivered to SB3, including any reward scaling or ``VecNormalize`` transformation.
Use a separate evaluation environment for task-reward units. When a step both terminates and truncates, SB3 retains
termination alone; ordinary ``evaluate()`` preserves both flags in its termination class.

Optuna
------

Install the ``optuna`` extra with ``python -m pip install '.[optuna]'`` from a checkout. ``OptunaReporter(trial)`` is an ``on_evaluation`` hook for the SB3 callback. It reports
``summary.mean_return`` by default, accepts a custom scalar objective, and raises Optuna's ``TrialPruned`` when the trial
requests pruning:

.. code-block:: python

   from npc_gym.integrations.optuna import OptunaReporter

   callback = SB3EvaluationCallback(
       evaluator=evaluate_model,
       evaluation_frequency=10_000,
       on_evaluation=OptunaReporter(trial),
       collect_training_episodes=False,
   )

Here ``trial`` is the current Optuna trial and ``evaluate_model`` is the evaluator
from the SB3 example above. Accessing an adapter without its extra raises an
``ImportError`` naming the missing dependency.
