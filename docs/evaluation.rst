Evaluation
==========

Evaluation measures a policy over complete episodes. It reports the rewards
returned by the environment, task-specific measurements, and any selected
monitor counts. Monitoring does not change the policy's actions or rewards.

Running an evaluation
---------------------

``evaluate(env, policy, *, episodes=1, monitors=None, seed=None, monitor_source="factories")``
returns an ``EvaluationSummary``. The policy must be callable as
``policy(observation, info) -> action``. It receives the current observation
and public information, including an action mask if the environment supplies one.

``episodes`` must be a positive integer. Each episode starts with ``reset`` and
ends when ``terminated`` or ``truncated`` is true. The seed applies to the first
reset only; later resets continue the same random stream. The caller owns the
environment and closes it after evaluation.

The **episode return** is the sum of rewards returned by ``env.step`` over the
episode, without discounting. The **episode length** is the number of steps;
reset contributes neither a reward nor a step. If the environment includes a
reward wrapper, its adjustments are included in the return. To measure task
rewards independently of training penalties or scaling, use a separate
evaluation environment without those transformations.

For example, evaluate a policy that chooses the first action allowed by Storm
Taxi's action mask, while recording the Emergency monitor:

.. code-block:: python

   from functools import partial
   import gymnasium as gym
   import npc_gym
   from npc_gym.evaluation import evaluate
   from npc_gym.monitors import make_builtin_monitor

   identifier = "taxi/emergency-v0"
   env = gym.make("npc_gym/StormTaxi-v0", max_episode_steps=100)
   try:
       summary = evaluate(
           env,
           lambda observation, info: next(i for i, allowed in enumerate(info["action_mask"]) if allowed),
           episodes=2,
           seed=0,
           monitors={identifier: partial(make_builtin_monitor, identifier)},
       )
       print(summary.mean_return, summary.mean_monitor_counts)
   finally:
       env.close()

Choosing a monitor source
-------------------------

Select one source explicitly:

.. list-table:: Monitor sources
   :header-rows: 1

   * - ``monitor_source``
     - Configuration
     - What the evaluator does
   * - ``"factories"`` (default)
     - Pass no-argument factories in ``monitors={name: factory}``.
     - Creates each monitor once per evaluation, resets it for each episode, and processes each input.
   * - ``"wrapper"``
     - Use ``MonitorWrapper`` on the environment and omit ``monitors``.
     - Reads the wrapper's existing occurrences and counts on reset and every step.

With no factories, the default mode records no monitor counts and requires no
labels. When factories are supplied, ``info["labels"]`` must be an iterable of
hashable labels; bare strings and bytes are rejected. Wrapper mode requires valid
``info["monitors"]`` snapshots. It does not run the monitors a second time.
Neither mode infers counts from restraining-bolt rewards or diagnostics.

See `monitor integration <monitors.rst#monitor-wrapper>`__ for the wrapper's
snapshot structure, and `SB3 recording <integrations.rst>`__ for vector environments.

Results and aggregation
-----------------------

An ``EpisodeResult`` describes one completed episode:

.. list-table:: Episode measurements
   :header-rows: 1

   * - Field
     - Meaning
   * - ``episode``
     - Episode index, starting at zero within this evaluation.
   * - ``episode_return``
     - Sum of the returned step rewards.
   * - ``length``
     - Number of environment steps.
   * - ``termination``
     - Task termination, external truncation, or both, as a ``TerminationClass`` value.
   * - ``metrics``
     - Final cumulative task measurements from ``info["episode_metrics"]``; empty if absent.
   * - ``monitor_counts``
     - Final counts indexed by configured monitor name and member name.

The evaluator copies counts after processing the final input and before the
next reset. A ``SimpleMonitor`` or ``ComplexMonitor`` contributes the member key
``"count"``. A ``MultiMonitor`` contributes its named members and explicit derived
counts. Counts must be nonnegative integers, excluding Booleans. Metric names,
monitor names, and member names must remain the same across episodes.

``EvaluationSummary`` retains all episode results and computes a mean and
population standard deviation for returns, lengths, each metric, and each named
count. Every episode has equal weight. For values *x₁, …, xₙ*, the mean is
*μ = (x₁ + … + xₙ) / n* and the standard deviation is
*√(((x₁ − μ)² + … + (xₙ − μ)²) / n)*. A single episode therefore has standard
deviation zero. Nested count statistics are exposed as ``mean_monitor_counts``
and ``std_monitor_counts``.

Saving results
--------------

Use writers as context managers. They refuse to replace an existing file unless
``overwrite=True``; the parent directory must already exist. The following
example saves ``summary`` from the evaluation above:

.. code-block:: python

   from npc_gym.evaluation import EpisodeCSVWriter, EvaluationJSONWriter

   with EpisodeCSVWriter("episodes.csv") as writer:
       writer.write_all(summary.episodes)
   with EvaluationJSONWriter("summary.json") as writer:
       writer.write(summary, metadata={"seed": 0, "return_units": "Taxi task reward"})

.. list-table:: Output formats
   :header-rows: 1

   * - Writer
     - Schema version
     - Contents
   * - ``EpisodeCSVWriter``
     - 4
     - One row per episode.
   * - ``LearningCurveCSVWriter``
     - 4
     - One aggregate evaluation point per row, indexed by actual training transitions.
   * - ``EvaluationJSONWriter``
     - 5
     - One document containing episodes, aggregates, and optional metadata.

CSV count columns use ``count/<escaped-monitor>/<escaped-member>``; curve columns
append ``_mean`` and ``_std``. Encode each path component independently with
``urllib.parse.quote(value, safe="")``. For example,
``count/taxi%2Femergency-v0/Emergency%20Violations``. Domain metrics use
``metric/<name>`` columns. The first row fixes the metric and count columns for
that file; later rows must use the same names.

JSON stores nested ``monitor_counts`` in both ``episodes`` and ``aggregate``.
An episode leaf is an integer; an aggregate leaf contains ``{"mean": ..., "std": ...}``.
Each ``EvaluationJSONWriter`` context accepts exactly one summary.

Learning curves
---------------

A learning curve records evaluation results at successive training timesteps.
At each chosen timestep, call ``evaluate`` on a separate evaluation environment
and pass its summary to ``LearningCurveCSVWriter.write(timesteps, summary)``.
The timestep is the number of training transitions completed, excluding evaluation.
This works with any learner whose policy can be called as
``policy(observation, info) -> action``; use a small adapter if necessary.
Your training loop or callback decides when to evaluate. Keep evaluation separate
from training state and random streams, and use consistent evaluation settings
across points. A final evaluation uses the same interface and can be saved separately.

``plot_learning_curve(source, output, *, counts=None, show_std=True, overwrite=False)``
plots recorded episode returns and monitor counts. Install ``npc-gym[plots]`` to
generate figures; evaluation and recording use only core dependencies.
``source`` is a CSV written by ``LearningCurveCSVWriter`` or a sequence of such
paths. No experiment metadata or training framework is required.
``output`` is a PNG, PDF or SVG path, whose parent directory must already exist.
The function returns that path and releases its figure; it opens no window and
does not change the plotting backend. Existing outputs require ``overwrite=True``.
Input files cannot be overwritten, output symlinks are refused, and failed
rendering leaves existing files intact.

By default, all named counts appear in a separate panel. To select counts, pass
``counts=[(monitor_name, member_name), ...]``; use ``counts=[]`` for returns alone.
Counts remain separate, including permissions and diagnostic measurements;
the plot does not add them into a violation total. Returns retain the recorded
reward units, so evaluate without training penalties or reward scaling when
you want task returns.

This complete example trains a tabular policy, records intermediate evaluations,
saves a final evaluation, and generates a plot. Run it in a directory where the
four output files do not already exist:

.. code-block:: python

   from contextlib import closing
   from functools import partial
   import gymnasium as gym
   import npc_gym
   from npc_gym.algorithms import TabularQLearning
   from npc_gym.evaluation import (
       EpisodeCSVWriter, EvaluationJSONWriter, LearningCurveCSVWriter,
       evaluate, plot_learning_curve,
   )
   from npc_gym.monitors import make_builtin_monitor

   monitors = {"emergency": partial(make_builtin_monitor, "taxi/emergency-v0")}
   with (
       closing(gym.make("npc_gym/StormTaxi-v0")) as train_env,
       closing(gym.make("npc_gym/StormTaxi-v0")) as eval_env,
   ):
       model = TabularQLearning(train_env, seed=0)
       with LearningCurveCSVWriter("learning_curve.csv") as curve:
           def record(timesteps):
               summary = evaluate(
                   eval_env, model, episodes=5, monitors=monitors, seed=10_000,
               )
               curve.write(timesteps, summary)

           def after_step(learner, step):
               if step.timesteps % 250 == 0:
                   record(step.timesteps)

           record(0)
           model.learn(1_000, callback=after_step)

       final = evaluate(eval_env, model, episodes=20, monitors=monitors, seed=20_000)
       with EpisodeCSVWriter("episodes.csv") as episodes:
           episodes.write_all(final.episodes)
       with EvaluationJSONWriter("final.json") as summary:
           summary.write(final)

   plot_learning_curve("learning_curve.csv", "learning_curve.png")

The short training budget illustrates the workflow, rather than establishing
learning quality. The callback belongs to this learner; the evaluation, writers
and plotting function are the same for other learning algorithms.

For one CSV, shading shows the recorded mean plus or minus one population
standard deviation across evaluation episodes. For several independent runs,
the plot averages their episode means with equal weight per run, and shading
shows the sample standard deviation across those run means. These bands are
not confidence intervals. Pass ``show_std=False`` to hide them.

To combine training seeds, for example::

   plot_learning_curve(
       ["seed-0.csv", "seed-1.csv", "seed-2.csv"], "across-seeds.pdf",
   )

Inputs must use the current curve schema, contain at least one evaluation point,
and have identical columns and strictly increasing, matching timestep grids.
Duplicate files, non-finite values, negative counts or standard deviations,
and unknown count selections are rejected. Each path should represent one
independent run. The caller must ensure comparable tasks, reward units, monitors
and evaluation protocols: a standalone CSV does not record that provenance.
The repository's `experiment tools <experiments.rst#plotting-results>`__ provide
metadata-based grouping when running the supplied experiment configurations.
