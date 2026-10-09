Experiments
===========

The repository's ``experiments`` runner combines versioned environments,
wrappers, norms and learning methods. It is separate from the installed
``npc_gym`` API; ``specifications.py`` defines its supported configurations.

Completed comparisons are presented in `experiment baselines <experiment_baselines.rst>`__.

All paper monitor and bolt configurations use minimized automata. Bolt state
features encode the minimized DFAs; norm definitions and punishment magnitudes
are specified below.

.. _paper-results:

Reproducing the paper results
-----------------------------

Use a repository checkout: the installed library does not include the runner
or retained data. Choose one of three workflows below.

Setup
~~~~~

For the recorded training versions::

   conda env create -f environment-paper.yml
   conda activate npcgym-paper

This pins Python 3.13.15, Gymnasium 1.3.0, NumPy 2.2.6, Stable-Baselines3 2.9.0,
Torch 2.14.1 and Clingo 5.8.2, and includes plotting and test tools.
``environment.yml`` is the development setup with version ranges.
For data analysis alone, install ``python -m pip install '.[plots]'`` in a
supported Python environment. Installation needs network access; reproduction
uses local data, runs on CPU and needs no display or MONA.

The retained data expand to about 5.2 GB after checkout. Allow additional space
for the environment and outputs. Use fresh directories under
``experiments/output/``; commands refuse to overwrite existing results.

1. Regenerate tables and figures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validate the data and generate the compact paper tables without checkpoints::

   python -m experiments.results --input experiments/results
   python -m experiments.generate_paper_tables --output experiments/output/paper-tables

The validator checks archive hashes, all recipes and seeds, completed runs and
1,000 final episodes with matching checkpoint identities. It writes nothing.
The table command writes two LaTeX tables and ``sources.json`` with unrounded
statistics and exact source paths. See `LaTeX paper tables <#latex-paper-tables>`__
for filenames and presentation options, or
`the result guide <experiment_baselines.rst#retained-data-and-reproduction>`__
for commands generating full reports and figures.

2. Verify execution with tiny runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the paper environment active::

   CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/experiments

These tests exercise training, saving, reloading, evaluation, export and
reporting using temporary outputs. They do not establish learning quality or
alter retained data. Optional rendering tests skip without rendering extras.

3. Retrain the paper configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``--paper`` with ``train`` or ``evaluate`` to select exactly the table
configurations: four Taxi recipes (five policies), four Merchant recipes, eight
Gardener recipes and seventeen Pacman recipes. Every recipe uses seeds 0--7
and 1,000 evaluation episodes. The paper selection fixes Pacman to Berkeley
``smallClassic`` with random ghosts, including its baseline and OFTEN recipes.
It excludes budget pilots and image experiments. Without ``--paper``, the
runner also exposes exploratory configurations.

Train all configurations in a fresh working directory, then run the named
paper evaluations::

   python -m experiments.run train --paper --output experiments/output/reproduction \
     --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --seed 6 --seed 7 --threads 1
   python -m experiments.run evaluate --paper --output experiments/output/reproduction \
     --name paper --episodes 1000 --current-monitors

Export the numerical data, repeating this command for ``taxi``, ``merchant``,
``gardener`` and ``pacman``::

   python -m experiments.run retain --input experiments/output/reproduction \
     --outdir experiments/output/paper-results --environment <environment>

Taxi retains ``final``;
the other environments retain the named ``paper`` evaluations. Then rerun the
validation and reporting commands on the exported data.

To select one environment, add ``--environment taxi/storm-v1``,
``merchant/basic-v2`` or ``gardener/size-15-v0``. Pacman needs both
``pacman/small-classic-v0`` and ``pacman/smallclassic-random-bolts-v0``.
For Linux parallel training, replace ``--threads 1`` with ``--cpus`` followed by
available CPU IDs; see `parallel execution <#pacman-smallclassic-restraining-bolts>`__.
Add ``--dry-run`` to inspect plans without starting jobs or creating output.

Pacman's paper selection applies all six norm bases to every policy. The consolidated table groups baselines, Vegan, Vegetarian Orange, Trapped,
HungryVegan, VeganConflict and HungryPenaltyVegan. It reports the six individual
norm counts on every row. Vegan and Vegetarian share one paired base; Trapped's
base remains separate because its observation includes the active restriction.
The paper includes no continued-DQN comparisons. OFTEN and its pretrained base
have different training budgets, which the tables report explicitly.

The paper Pacman OFTEN recipes are ``pacman-dqn-often-v2``,
``pacman-dqn-often-vegetarian-v2`` and ``pacman-dqn-often-trapped-v3``. Each uses
5M-step γ=0.99 base training with a 300-step limit, then an adapted budget of
10M combined teaching transitions with a 500-step limit. All evaluations,
including fixes and teaching curves, use 300 steps. Paired evaluations use
episode seeds 50,000–50,999 for every training seed. Vegan and Vegetarian share
the main DQN base networks; Trapped trains a separate base with its
restriction-active observation. The paper report omits the redundant shared
base row. Other DQN settings are SB3 defaults, including exploration fraction
0.1 and five native actions.
The base-training evaluation protocol uses one initial seed of 10,000 plus the
training seed, evaluated every 250k steps. Teaching is evaluated every 1M
transitions. Retained source diagnostics identify their original evaluation
protocol; the selected final measurements and reproduction use 300 steps.

Gardener's paper selection monitors Collect One, per-frog Rescue, unpermitted
collection, Drain and No Collect on every policy. Its consolidated table includes
three baselines, four fixes, four OFTEN policies and two bolt policies:

Every retained final comparison uses 1,000 evaluation episodes per seed.
Reproduce learner settings, seeds and software versions from ``run.json``
when comparing numbers. Exact training outputs can depend on library and
hardware versions; matching settings does not promise bit-for-bit training
reproduction. The retained runs record approximately 153 summed training hours
for Pacman, 2.3 for Gardener, 0.7 for Taxi and 0.6 for Merchant on the original
host. These are recorded training durations, not an estimate of total rerun
wall time: concurrency, evaluations and hardware affect the actual cost.

The original training scenarios and diagnostic schedules remain in the run
metadata. Named ``paper`` evaluations can add current monitors without changing
those training settings. Compare final measurements using the selected
evaluation protocol, rather than assuming every diagnostic used that protocol.

Selecting and running a configuration
-------------------------------------

The command-line runner requires an output root and explicit training seeds. Inspect fully resolved plans without
training or writing files:

.. code-block:: console

   python experiments/run.py train --output experiments/output --seed 0 --dry-run

Train one supported configuration, then run another named evaluation of its saved checkpoint:

.. code-block:: console

   python experiments/run.py train --output experiments/output \
     --experiment taxi-tabular-unconstrained-v1 --seed 0
   python experiments/run.py evaluate --output experiments/output \
     --experiment taxi-tabular-unconstrained-v1 --seed 0 --name confirmation

The runner validates the complete selected batch before training. Existing output, incompatible identities, unsafe
path components, missing extras, and path collisions fail before work begins. ``--overwrite`` applies only to matching
run-owned training output or the selected named evaluation; evaluations never overwrite models.

Named evaluation defaults to the current recipe's episode count (1,000 for the
paper experiments); ``--episodes`` explicitly overrides it. Add
``--current-monitors`` to evaluate a saved checkpoint with the current recipe's
full monitor set. This changes only evaluation monitoring: saved environment,
learner and observation settings are retained, and the source run and checkpoint
are unchanged. Each evaluation records its monitor IDs and checkpoint hash.
Named OFTEN evaluations include the taught policy, the pretrained base in
``base/``, and the base with policy fixes in ``base-fixed/``, all using the same
episode seeds and monitor set.

Add ``--render`` to ``evaluate`` to watch automatically presented human frames. Taxi uses 4 FPS, Gardener and Pacman
use 10 FPS, and the Merchant baseline uses a 500 ms frame delay. Headless evaluation has no rendering delay.

Unconstrained baselines
-----------------------

These configurations learn task rewards without normative penalties or policy
fixes. Monitors report violations during evaluation only.

.. list-table:: Baseline configurations
   :header-rows: 1

   * - Experiment
     - Steps
     - Learning settings
   * - ``taxi-tabular-unconstrained-v1``
     - 5M
     - Taxi tabular Q-learning baseline with compact observations.
   * - ``merchant-tabular-unconstrained-v2``
     - 5M
     - Merchant tabular Q-learning baseline.
   * - ``gardener-dqn-unconstrained-v0``
     - 100k
     - Gardener DQN baseline.
   * - ``gardener-ppo-unconstrained-v0``
     - 100k
     - SB3 PPO defaults.
   * - ``pacman-ppo-unconstrained-v2``
     - 5M
     - SB3 PPO defaults.
   * - ``pacman-dqn-unconstrained-v1``
     - 5M
     - SB3 DQN defaults.
   * - ``pacman-images-dqn-unconstrained-v1``
     - 5M
     - Image-example DQN settings, without bolts.

Install ``.[sb3]`` for DQN/PPO or ``.[sb3,render]`` for image DQN. All run on
CPU. Gardener PPO and feature-based Pacman DQN use discount 0.99 and default
network architectures. PPO completes whole rollouts, so its actual step count
can exceed the requested budget. Gardener uses state features and maps illegal
actions to Stay with a training penalty of -1. Pacman uses ``small``
and divides training rewards by 100; evaluation reports raw task rewards.

Image DQN uses two 80-by-210 grayscale frames, a NatureCNN with 128 output
features and a 64–64 Q-network head. Its replay buffer holds 100,000 transitions
(about 6.72 GB of pixels); other DQN learning settings are defaults. It has no
automaton features or norm rewards. Episodes are limited to 300 Pacman steps or
1,000 Gardener steps. All registered experiments use 1,000 evaluation episodes
per training seed, both at intermediate checkpoints and for final evaluations.
Paired methods share the same 1,000 episode seeds within each comparison.

For example:

.. code-block:: console

   python experiments/run.py train --output experiments/output \
     --experiment pacman-images-dqn-unconstrained-v1 --seed 0

Pacman smallClassic restraining bolts
-------------------------------------

These six norm bases require a repository checkout containing
``experiments/layouts/smallClassic.lay`` and the ``sb3`` extra. The layout is not
part of installed distributions; its checksum is validated
before training. They use random ghosts, complete high-level features and a
300-turn limit. DQN and PPO start from scratch with fixed SB3 2.9 default
learning settings, including discount 0.99 and the default MLP architecture.
All bolt DFAs are minimized.

.. list-table:: Budgets and raw punishment magnitudes
   :header-rows: 1

   * - Norm key
     - DQN steps
     - PPO steps
     - Punishments
   * - ``vegan``
     - 2.5M
     - 5M
     - Eating either ghost: 1636.632265720805
   * - ``vegetarian``
     - 10M
     - 20M
     - Eating orange: 3710.6783688206015; blue is permitted
   * - ``hungry-vegan``
     - 20M
     - 20M
     - Hungry: 4719.7584507181455; eating either ghost: 714.172186332458
   * - ``trapped``
     - 20M
     - 20M
     - Each forbidden-zone action while restricted: 1000
   * - ``vegan-conflict``
     - 5M
     - 5M
     - Eating either ghost: 1000; missed blue obligation: 3000
   * - ``hungry-vegan-penalty``
     - 20M
     - 20M
     - Hungry: 3067.2038106118816; eating either ghost: 426.1333673762767;
       failure to pause after eating: 806.9335974146014

The experiment IDs are ``pacman-smallclassic-<dqn|ppo>-bolts-<norm-key>-v0``;
Trapped uses ``trapped-20m`` and VeganConflict uses ``vegan-conflict-5m`` in
place of the norm key. Reward-wrapper IDs end in ``bolts-and-reward-v1``.

Observations prepend the active bolts' separate one-hot state vectors to the
unchanged 63 game features. Trapped alone appends a restriction-active flag to
those game features. The one-hot feature count is the sum of the component DFA
sizes, not the number of combinations of component states.

.. list-table:: Minimized bolt observations
   :header-rows: 1

   * - Norm base
     - Component DFA sizes
     - One-hot features
     - Total observation features
   * - Vegan
     - 2 + 2
     - 4
     - 67
   * - Vegetarian Orange
     - 2
     - 2
     - 65
   * - HungryVegan
     - 2 + 2 + 3
     - 7
     - 70
   * - Trapped
     - 3
     - 3
     - 67
   * - VeganConflict
     - 2 + 2 + 5
     - 9
     - 72
   * - HungryPenaltyVegan
     - 2 + 2 + 3 + 4
     - 11
     - 74

Training receives ``(task reward - punishments) / 100``. Evaluation retains
the same observations but reports unscaled task returns and separate monitor
counts for all six norm bases. The `norm definitions <catalogue/pacman.rst>`__
apply, including Hungry's score-greater-than-100/loss deadline and satisfaction
by eating on that input. HungryPenaltyVegan uses the ``penalty1`` weights of
`Neufeld, Engesser, and Tappler <https://doi.org/10.24963/KR.2026/99>`__;
VeganConflict uses its reference weights; Trapped uses a 1000-point punishment.

Run all 96 learner/seed combinations through the central paper runner:

.. code-block:: console

   python experiments/run.py train --paper \
     --output experiments/output/pacman-bolts-reproduction \
     --environment pacman/smallclassic-random-bolts-v0 \
     --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --seed 6 --seed 7 \
     --cpus 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
   python experiments/run.py evaluate --paper \
     --output experiments/output/pacman-bolts-reproduction \
     --environment pacman/smallclassic-random-bolts-v0 \
     --name paper --episodes 1000 --current-monitors

``--cpus`` requires Linux and ``taskset``. It runs one single-threaded process
per listed CPU, queues longer runs first, and retains ``batch.json`` and
``batch-logs/``. Choose distinct physical cores on your machine and a fresh
output directory. Omit ``--cpus`` and add ``--threads 1`` for sequential
training. Add ``--dry-run`` to inspect the plans without creating output.
Use ``--experiment`` to select a single configuration.

Every seed evaluates 1,000 episodes before training, at 20 milestones, and after
reloading the final checkpoint. Intermediate evaluation starts with seed
``10000 + training seed``; final and named paper evaluations start with seed
``10001 + training seed``. Later episodes continue that RNG stream. PPO finishes
its final rollout and can slightly exceed the requested budget.

The `paper workflow <#paper-results>`__ also trains baselines, policy fixes and
OFTEN, and exports exactly the numerical file layout in
``experiments/results/pacman``. It retains all seeds, run metadata, checkpoint
hashes, learning curves and intermediate evaluations. Checkpoints, replay
buffers and logs remain in ignored working output. Budget pilots and image
experiments are excluded from the paper selection.

Taxi comparisons
----------------

Taxi has one compact-observation baseline and exactly five compared policies.
All four training recipes use tabular Q-learning with learning rate 0.2, discount
0.99, epsilon decreasing from 1.0 to 0.1 over the first 10% of training,
no action masks, and a 50-step episode limit. Baseline/fixes train for 5M steps,
Warn-only bolts for 10M and Emergency bolts for 50M. All use the default
exploration fraction 0.1: decay spans 500k, 1M and 5M steps respectively.
Location labels
describe the actual position after the action. Every recipe monitors all Emergency
components: Warn, Stay, Seven-step safety and Three-step safety, plus the
Safety and Emergency totals. Every Taxi monitor and bolt uses minimized DFAs.

.. list-table:: Five policies from four training recipes
   :header-rows: 1

   * - Policy
     - Experiment ID
     - Observation and objective
   * - Compact baseline
     - ``taxi-tabular-unconstrained-v1``
     - Compact state; task reward only. This is the reference for every comparison.
   * - Base for policy fixes
     - ``taxi-tabular-policy-fixes-warn-v1`` (base evaluation)
     - Compact state; task reward only.
   * - Policy fixes
     - ``taxi-tabular-policy-fixes-warn-v1`` (fixed evaluation)
     - Same base policy; the fixer tracks rain history and optimizes Warn only.
   * - Warn-only bolt
     - ``taxi-tabular-bolts-warn-v5``
     - Full state and one minimized automaton; penalty 5 for Warn.
   * - Emergency bolts
     - ``taxi-tabular-bolts-emergency-50m-v1``
     - Full state and four minimized automata; penalty 5 for each component violation.

Install ``python -m pip install '.[asp]'`` to include the policy-fix comparison.
Run all eight seeds for the four recipes with one worker per physical CPU:

.. code-block:: console

   python experiments/run.py train --paper --output experiments/output/taxi-paper \
     --environment taxi/storm-v1 \
     --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --seed 6 --seed 7 \
     --cpus 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

Use a fresh output directory. To train just the bolts while reusing completed
baseline/fix results, replace ``--environment taxi/storm-v1`` with
``--experiment taxi-tabular-bolts-warn-v5 --experiment taxi-tabular-bolts-emergency-50m-v1``.

The `included results <experiment_baselines.rst>`__ use these recipes and the
default exploration fraction 0.1 for every Taxi policy.

The ``taxi-emergency`` report group compares all five policies and all norm counts:

.. code-block:: console

   python experiments/run.py report \
     --input experiments/output/taxi-paper \
     --outdir experiments/output/taxi-paper-report \
     --benchmark taxi-emergency

Taxi Emergency restraining bolts
---------------------------------

The `Emergency norm base <catalogue/taxi.rst#catalogue-emergency>`__ retains its
rain-history-dependent safety deadlines. Each of the four component violations
subtracts 5 reward units. Simultaneous penalties add; the derived Safety and
Emergency totals incur no additional penalty.

The learner observes the full ``Discrete(352000)`` environment state, including
home, flood risk and shelter, plus four minimized bolt automaton states:
Warn has 5 states, Stay 3, Seven-step safety 18 and Three-step safety 6.
``IgnoreWeatherRelevant`` is not used by the bolts. Training rewards are unscaled.
Evaluation returns the unadjusted task reward while preserving the bolt observations.

The baseline and both bolt recipes evaluate 1,000 episodes initially, every
250k training steps, and after reloading the final checkpoint. Their RNG seeds
are 10000 + training seed for intermediate evaluations and 10001 + training
seed for the final evaluation. Completed numerical results belong in
``experiments/results/taxi/`` and are reported in
`the result table <experiment_baselines.rst#taxi-emergency-and-warn>`__.

Taxi Warn restraining bolt
--------------------------

The Warn-only bolt trains for 10M steps, with default exploration decay over
the first 1M steps. The remaining learner parameters, episode limit and
evaluation schedule match the Emergency bolts. Its observation adds only the minimized
five-state Warn automaton to the full environment state. It subtracts 5 reward
units for each missed rain-onset response. Safety and Stay are monitored but
have no effect on training rewards.

Taxi Warn policy fixes
----------------------

Install ``python -m pip install '.[asp]'``. Train a norm-free Storm Taxi Q-table
for 5 million steps, then compare it with and without the one-step Warn fix:

.. code-block:: console

   python experiments/run.py train --output experiments/output \
     --experiment taxi-tabular-policy-fixes-warn-v1 \
     --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --seed 6 --seed 7

Learning uses rate 0.2, discount factor 0.99 and epsilon decreasing from 1 to 0.1
over the first 500,000 steps. ``IgnoreWeatherRelevant`` supplies compact
observations; episodes last at most 50 steps. Rewards are unchanged and neither
training nor evaluation uses action masks. The fixer tracks rain onset and
prioritizes warning on the next step, then policy preference rank. Solver reuse
is enabled; history resets with each episode.

Every training seed uses the same 1,000 paired evaluation seeds, 10000–10999.
``model.zip`` contains the base Q-table. ``evaluations/final`` contains fixed-policy
results, with base results in its ``base`` subdirectory. Outputs include task
return, delivery success, episode length and all Emergency counts; only Warn is
enforced. Metadata records interventions, solver fallbacks and selection time,
while ``run.json`` records training time excluding evaluations. The learning
curve evaluates the fixed policy before and after training. Named ``evaluate``
runs repeat both variants.

Merchant Environment Friendly policy fixes
------------------------------------------

Install ``python -m pip install '.[asp]'``. This configuration trains a norm-free,
action-masked Q-table for 5 million steps, then compares the saved policy with
and without the one-step Environment Friendly fix:

.. code-block:: console

   python experiments/run.py train --output experiments/output \
     --experiment merchant-tabular-policy-fixes-env-friendly-v0 \
     --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --seed 6 --seed 7

Learning uses rate 0.5, discount factor 0.99 and epsilon decreasing from 1 to 0.2
over the first 500,000 steps. The environment is ``basic`` Merchant with capacity
5, attack probability 0.75, death probability 0.25, sunset 28 and a 150-step limit.
``IgnoreTimeObservation`` omits the clock; task rewards are unchanged. Both
variants respect the action mask. The fixer first minimizes prohibited extraction
attempts, then policy preference rank, with solver reuse enabled.

Every training seed uses the same 1,000 paired evaluation seeds, 10000–10999.
``model.zip`` contains the unchanged base Q-table. ``evaluations/final`` contains
fixed-policy results; its ``base`` subdirectory contains the matching base results.
Both report raw task return, episode length, task outcomes and all paper norms:
Environment Friendly, Delivery, Danger and CTD, plus Pacifist and DeliveryPacifist
totals. Only Environment Friendly is enforced. Fixed-policy
metadata includes interventions, solver fallbacks and policy-selection time;
``run.json`` records training time excluding evaluations.

The learning curve evaluates the fixed policy before and after training. Named
``evaluate`` runs repeat both variants in the same directory structure. Norm
adherence and task return should be read separately: limiting wood collection
also removes opportunities to earn collection and delivery rewards.

Merchant restraining bolts
--------------------------

The two Merchant bolt configurations use the same action-masked tabular learner,
environment parameters and 150-step episode limit as the Merchant baseline.
Environment Friendly omits the clock as in that baseline; DeliveryPacifist
retains the native clock so the learner observes time remaining before sundown.
Each adds the active norm base's minimized discrete bolt
states to the observation: 4 for Environment Friendly, and 3 Delivery, 2 Danger
and 4 CTD states for DeliveryPacifist. Environment Friendly penalizes each prohibited extraction
by 300. DeliveryPacifist penalizes each missed delivery by 10,000, each Danger
violation by 100 and each CTD (fighting) violation by 1,000. These penalties add
on simultaneous events; training rewards are not scaled. Evaluation returns the
original task reward.

Run both configurations for 5 million steps per seed:

.. code-block:: console

   python experiments/run.py train --output experiments/output/merchant-bolts-minimized \
     --experiment merchant-tabular-bolts-env-friendly-minimized-v0 \
     --experiment merchant-tabular-bolts-delivery-pacifist-minimized-v1 \
     --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --seed 6 --seed 7 \
     --cpus 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

Choose available CPUs for your machine. The runner queues 16 runs with one
single-threaded worker per listed CPU. Learning uses rate 0.5, discount 0.99,
and epsilon decreasing from 1 to 0.2 over the first 500,000 steps. Learning curves
evaluate 1,000 episodes before training and every 250,000 steps. Final evaluation
reloads each checkpoint and uses 1,000 episodes. For training seed ``s``, evaluation
RNGs start at ``10000 + s`` for intermediate evaluations and ``10001 + s`` for
final evaluation, matching the baseline runner's seed convention.

Both configurations report Environment Friendly, Delivery, Pacifist and
DeliveryPacifist counts. Only the selected base contributes penalties and bolt
observations. Evolving is excluded. DeliveryPacifist's observation includes the
clock at tuple index 5 alongside its bolt states; the other three Merchant
policies omit the clock. The clock starts at zero, advances on every action,
and remains at 28 from sundown onward.
Numerical results, learning curves and provenance are saved with each run;
checkpoints and process logs remain local.

The four Merchant policies share one ``merchant`` comparison: the shared
baseline, Environment Friendly fixes, Environment Friendly bolts and
DeliveryPacifist bolts. The policy-fix recipe uses the baseline learner and
observations; its raw-base evaluation stays in working output and does not
produce another retained result or table row. Each uses 5M training steps and eight seeds
(0--7). Increased training budgets are deferred. Every method evaluates 1,000
episodes and monitors all paper norms; Evolving is excluded.

Train exactly these four recipes for all eight seeds in a fresh output root:

.. code-block:: console

   python experiments/run.py train --output experiments/output/merchant \
     --environment merchant/basic-v2 \
     --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --seed 6 --seed 7 \
     --cpus 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
   python experiments/run.py report --input experiments/output/merchant \
     --outdir experiments/output/merchant-report --benchmark merchant

The environment filter selects exactly the four paper recipes (32 training
runs). The report produces four rows and includes every seed. Each recipe
retains its specified evaluation seed convention and episode limit.

To reevaluate saved checkpoints, place the four complete experiment directories
with their checkpoints under one output root, then run:

.. code-block:: console

   python experiments/run.py evaluate --output experiments/output/merchant \
     --experiment merchant-tabular-unconstrained-v2 \
     --experiment merchant-tabular-policy-fixes-env-friendly-v0 \
     --experiment merchant-tabular-bolts-env-friendly-minimized-v0 \
     --experiment merchant-tabular-bolts-delivery-pacifist-minimized-v1 \
     --name paper --episodes 1000 --current-monitors

Omitting ``--seed`` selects all saved seeds. The fixed and raw base policies
use the same episode seeds. Retained complete evaluations are in
``experiments/results/merchant/``. Regenerate the
combined table without checkpoints:

.. code-block:: console

   python experiments/run.py report \
     --input experiments/results/merchant \
     --outdir experiments/output/merchant-paper-report \
     --benchmark merchant --evaluation-name paper

The named report uses only the selected final evaluations, with one point per
policy and seed. It does not combine these measurements with training curves.

Pacman DQN and OFTEN
---------------------

Install ``python -m pip install '.[sb3,asp]'`` and select a configuration:

.. list-table:: Pacman OFTEN configurations
   :header-rows: 1

   * - Experiment
     - Norm
   * - ``pacman-dqn-often-v1``
     - Both blue and orange (Vegan).
   * - ``pacman-dqn-often-vegetarian-v1``
     - Orange only (Vegetarian Orange, matching the reference's Vegetarian norm).
   * - ``pacman-dqn-often-trapped-v2``
     - Stay west from score zero until the actual game score exceeds 400.

Each seed trains its own DQN for 5 million steps, saves and reloads it, then runs
10 million OFTEN transitions: 5 million ordinary and 5 million expert.
Both phases use discount factor 0.99, rewards divided by 100 and a 64–64 network;
other DQN learning settings are SB3 defaults. OFTEN uses margin 0.5, its default
expert-weight schedule, and a one-step planner. Vegan and Vegetarian use radius
five; Trapped uses exact player movement and observed restriction history.
The environment is ``small`` with complete feature
observations. Base training and all evaluations use a 300-step limit; teaching
uses a 500-step limit.

Run the eight training seeds, or add ``--dry-run`` to inspect their plans:

.. code-block:: console

   python experiments/run.py train --output experiments/output \
     --experiment pacman-dqn-often-v1 \
     --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --seed 6 --seed 7

Select ``pacman-dqn-often-vegetarian-v1`` or ``pacman-dqn-often-trapped-v2``
instead, or repeat ``--experiment`` to run multiple configurations. Each uses its matching policy fixes
and monitor; their other settings and evaluation seeds are identical.

Trapped's planner updates its history separately in the ordinary and expert
environments. It does not predict score changes; see its
`abstraction contract <catalogue/pacman.rst#trapped-policy-fix>`__.
``TrappedObservation`` appends a restriction-active flag to the complete
observations in base training, teaching and evaluation. This exposes the history
needed to distinguish an active restriction from one already released. Its monitor,
planner and observation flag all release at score greater than 400; score 400
still requires staying west. Training reward scaling does not affect this threshold.

The runner executes seeds sequentially, using one CPU thread for each training
run. Teaching uses seed 20000 plus the base seed. The base, every million teaching
transitions and the final saved policy are evaluated on the same 1,000 episode
seeds, 50000–50999, with raw task rewards and no planner.

``base.zip`` contains the pretrained DQN; ``model.zip`` contains the taught policy.
``run.json`` records both checkpoint hashes, separate phase timings and teaching
counts. Teaching time excludes intermediate evaluation. The learning curve starts
at the pretrained base (zero teaching transitions); its horizontal axis counts
ordinary and expert transitions together, excluding pretraining.
``base_learning_curve.csv`` and ``evaluations/base/intermediate-<steps>/`` record
pretraining separately, at zero, each evaluation interval and the final base. Per-episode
outputs contain win status, task return and the selected norm's violations.
The runner also evaluates the base with policy fixes, as described below.

Gardener DQN and OFTEN
----------------------

The Gardener baseline and OFTEN configurations monitor three norms: collection
is prohibited unless permitted, harmful drainage is prohibited, and Rescue
requires timely frog collection after harmful drainage. The OFTEN variants
differ only in their teaching targets:

.. list-table:: Gardener OFTEN configurations
   :header-rows: 1

   * - Experiment
     - Prohibited action
   * - ``gardener-dqn-often-no-collect-v0``
     - Collecting any frog.
   * - ``gardener-dqn-often-drain-v0``
     - Draining a puddle near a surviving frog.
   * - ``gardener-dqn-often-permission-v0``
     - Collecting a frog without a legal move right.
   * - ``gardener-dqn-often-permission-drain-v0``
     - Unpermitted collection and harmful drainage together.

The combined configuration treats permission as an exception to No Collect.
These Gardener experiments report ``Unpermitted`` collection, ``Drain`` and
``Rescue`` failures, including the unconstrained baselines. ``Collected`` and
``Permitted`` remain descriptive counts. Policy fixes and OFTEN do not optimize
Rescue. The separate Collect One monitor is not part of this norm base.

Each seed trains a fresh DQN for 250,000 steps, then teaches it for 250,000 combined
transitions: 125,000 ordinary and 125,000 expert. Both phases use discount factor
0.95 and a 64–64 network. Pretraining uses the other SB3 DQN defaults. Teaching
changes only ``batch_size=256`` and ``gradient_steps=-1``: four learning updates
after each four ordinary plus four expert transitions. OFTEN uses margin 5,
expert weight increasing from 0.1 to 0.5, replay filtering, and a one-action planner
with radius four.

The environment is size 15, with score limit 300 and a 1,000-step episode limit.
``StateFeatureObsWrapper(include_frogs=True)`` supplies 27 features, including
collection/drainage risks and action legality, during both phases. Rewards are
not scaled; illegal actions become Stay with an additional training penalty of
−1. Evaluation omits that penalty; the taught policy acts without a planner.

Run three training seeds with the same ``sb3,asp`` extras:

.. code-block:: console

   python experiments/run.py train --output experiments/output \
     --experiment gardener-dqn-often-no-collect-v0 \
     --experiment gardener-dqn-often-drain-v0 \
     --experiment gardener-dqn-often-permission-v0 \
     --experiment gardener-dqn-often-permission-drain-v0 \
     --seed 0 --seed 1 --seed 2

The runner saves ``base.zip`` and ``model.zip``, with separate phase timings and
teaching counters as for Pacman. Teaching uses seed 20000 plus the base seed.
Base and taught policies are evaluated on the same 1,000 episode seeds,
10000–10999, reporting task return, final score, episode length and the selected
norms' counts. The taught policy acts without a planner, alongside a separate
base-with-fixes comparison. There is no continued-DQN control.

OFTEN comparisons
-----------------

Every OFTEN evaluation includes the saved base policy with the same policy fixes
used for teaching: the same norm recipe, objectives, horizon, radius and state
updates. This comparison uses greedy base actions corrected by the planner,
without exploration or further learning. Both policies use the same episode
seeds, limits, observations, raw task rewards and monitors.

Each ``evaluations/<name>/`` directory holds ``summary.json`` and ``episodes.csv``
for the evaluated policy, plus the same files under ``base-fixed/`` for the
corrected base. Both report task performance and all monitor counts. The
comparison metadata identifies the base checkpoint and planner settings, and
records interventions, solver fallbacks, policy-selection time and evaluation
time. It does not alter the checkpoints or training random state.

Intermediate points reuse one measured base-with-fixes reference because that
policy and its evaluation seeds are fixed. Final and named evaluations measure
both policies, honoring any ``--episodes`` or ``--evaluation-seed`` overrides.
``learning_curve.csv`` follows the policy being taught; the reference results
remain in each point's ``base-fixed/`` directory. The experiment runner's
``evaluate`` command requires the ``sb3,asp`` extras for this paired comparison;
using the taught policy alone requires no planner.

Run outputs
-----------

Each seed directory contains ``run.json``, a staged and verified model, intermediate evaluation directories, and a final
evaluation. Metadata records resolved configuration, component and wrapper order, seeds, requested and actual
timesteps, package and Git provenance, device, task-return units, checksums, and completion status. Failed runs remain
explicitly incomplete rather than being silently resumed.

Retaining benchmark data
------------------------

Use ``experiments/output/`` for checkpoints, replay buffers, process logs and
working runs. The ``retain`` command copies only the selected paper recipes to
``experiments/results/<environment>/<experiment-id>/seed-<seed>/``. It requires
all eight seeds (0--7), completed ``run.json`` records, final episode results,
matching checkpoint hashes and learning curves. Missing or incomplete runs,
duplicate sources, overlapping paths and existing destinations are rejected.
Files are copied unchanged and listed with SHA-256 hashes in ``archive.json``;
failed publication removes its temporary output. No source files are modified.
Repeat ``--input`` for configurations trained in separate output roots. Use
``--experiment`` to retain a complete subset while other configurations remain
pending; the manifest lists exactly what is present. Do not select seeds by
performance. Pilots and superseded results do not belong in the paper dataset.

Merchant retains its named ``paper`` evaluations and excludes the duplicate raw
base. Other environments retain final, intermediate and OFTEN base evaluations.
Unrelated named evaluations, checkpoints and logs are excluded. Training curves
and recorded source configurations stay unchanged when final policies are
reevaluated with additional monitors.

The numerical results can be analyzed and plotted without checkpoints:

.. code-block:: console

   python experiments/plot_learning_curves.py \
     --input experiments/results/<environment> --outdir plots

Replace ``<environment>`` with ``taxi``, ``merchant``, ``gardener`` or ``pacman``.
Running fresh evaluations requires the original local run and its checkpoints; keep those under
``experiments/output/`` or in separate model storage.

Evaluation schedule
-------------------

Single-stage learning curves start with one untrained-policy evaluation at timestep zero, followed by the configured transition
intervals. Each evaluation restarts from the intermediate evaluation seed. A vector step that crosses several intervals
produces one curve row and one directory at its actual timestep; JSON metadata lists all covered thresholds in
``scheduled_thresholds``. The initial list is ``[0]``.
SB3 evaluations restore training random state and policy modes even on failure; tabular prediction uses a separate
random stream from learning. The final evaluation reloads the saved checkpoint and uses its own seed and checksum.
It remains separate because SB3 can optimize the policy after its last intermediate evaluation.

The training CLI enables INFO progress messages on stderr, including the tabular learner's configured logging interval.
Importing the runner does not configure logging.

Plotting results
----------------

``plot_learning_curves.py`` discovers complete runs through metadata, validates compatible schemas and units, groups
the full resolved configuration across seeds, and plots unscaled task returns with between-seed standard deviation:

.. code-block:: console

   python experiments/plot_learning_curves.py --input experiments/output --outdir plots

Plots require the schema-4 learning curves described in `evaluation outputs <evaluation.rst#saving-results>`__.
Legends identify named monitor counts, including explicit derived totals.
Invalid inputs and protected output paths produce a concise error on stderr and exit status 2. Existing plots require
``--overwrite``; unsafe paths and colliding output names are refused even with that flag.

Gardener collection and rescue bolts
------------------------------------------------

The bolt norm base combines Collect One, per-frog Rescue, permission-aware
collection and Drain. It uses the size-15 environment, score target 300 and a
1,000-step limit. Training subtracts these penalties without reward scaling:

.. list-table:: Gardener bolt penalties
   :header-rows: 1

   * - Norm
     - Penalty
     - Counting unit
   * - Collect One
     - 10,000
     - Once at episode end if no frog was collected.
   * - Rescue
     - 1,000
     - Each frog with at least one failed obligation on that step.
   * - Permission-aware collection
     - 100
     - Each frog collected without permission.
   * - Drain
     - 10
     - Each puddle drained near an uncollected frog.

Rescue uses ``gardener/rescue-v2``: two frogs failing together cost 2,000.
Repeated obligations for the same frog failing together cost 1,000. Moving
Right permits collection; satisfying Rescue or Collect One grants no additional
exemption. The weights express soft priorities under discounted return.
Collect One's terminal penalty may encourage postponement when discounted;
inspect both its violation count and the timeout rate when assessing learning.

Each DQN/PPO run uses 1M requested training steps, SB3 defaults and discount
factor 0.99 on CPU. PPO completes its last rollout and can exceed the requested
budget. Eight training seeds are 0--7. Evaluate 1,000 deterministic episodes at
step zero and every 50,000 training steps, then 1,000 episodes after reloading
the final checkpoint. Evaluation seeds are 10,000 plus the training seed for
intermediate evaluations and 10,001 plus the training seed for the final one.

Observations contain 298 values: lexically sorted one-hot automaton features
precede the 27 frog-aware task features. All automata are minimized. These task features
describe immediate risks, not individual frog locations or routes to distant
frogs. Training retains the additional penalty of 1 for illegal proposals;
evaluation reports raw task returns without either action or norm penalties.

.. code-block:: console

   python experiments/run.py train --output experiments/output/gardener-bolts \
     --experiment gardener-dqn-bolts-collection-rescue-1m-v0 \
     --experiment gardener-ppo-bolts-collection-rescue-1m-v0 \
     --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --seed 6 --seed 7 \
     --cpus 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

The paper's ``gardener-paper`` report compares all policies with the same five
individual norm monitors and 1,000 episodes per seed, including per-frog Rescue.
All 64 completed training runs are retained in ``experiments/results/gardener/``
with learning curves, intermediate evaluations, named ``paper`` evaluations,
configurations and checkpoint hashes. Checkpoints and process logs remain local.
The `paper reproduction commands <#paper-results>`__ train, evaluate and retain
the same configurations and numerical file layout.

Comparison tables and curves
----------------------------

The report command compares methods within each environment and norm base.
It reads completed runs, including their episode results, without checkpoints.
Install the ``plots`` extra, then write to a new directory outside the input tree:

.. code-block:: console

   python experiments/run.py report --input experiments/results/merchant \
     --outdir reports/merchant --evaluation-name paper

``index.md`` links to one comparison per available benchmark. Each comparison
contains ``table.md`` and ``table.csv``, return and violation figures in PNG/PDF,
``curves.csv``, and ``sources.json`` with exact configurations and source paths.
Use ``--benchmark <id>`` to select a comparison; repeat it for several.
``--evaluation-name <name>`` reports named final evaluations instead of the
training run's ``final`` results and intermediate curves.

.. list-table:: Comparison groups
   :header-rows: 1

   * - ID
     - Norm base and methods
   * - ``taxi-emergency``
     - All Emergency counts; compact baseline, policy-fix base, Warn fixes, Warn-only bolt and Emergency bolts.
   * - ``merchant``
     - All paper norms; shared baseline, Environment Friendly fixes and bolts, and DeliveryPacifist bolts.
   * - ``gardener-paper``
     - All paper policies with the five individual norms; select explicitly with ``--evaluation-name paper``.
   * - ``gardener``
     - Permission-aware collection, Drain and Rescue; DQN/PPO baselines and the four fix/OFTEN variants.
   * - ``gardener-collection-rescue``
     - Collect One, per-frog Rescue, permission-aware collection and Drain; DQN/PPO bolts.
   * - ``pacman-paper``
     - All six individual paper norms, grouped baselines, fixes, OFTEN and DQN/PPO bolts; select explicitly with ``--evaluation-name paper``.
   * - ``pacman-vegan``
     - Small map, Vegan; DQN/PPO baselines and matching fixes/OFTEN.
   * - ``pacman-vegetarian``
     - Small map, Vegetarian Orange; DQN/PPO baselines and matching fixes/OFTEN.
   * - ``pacman-trapped``
     - Small map, Trapped; DQN/PPO baselines and matching fixes/OFTEN.
   * - ``pacman-smallclassic-vegan``
     - Paper smallClassic, Vegan; DQN/PPO baselines and matching fixes/OFTEN.
   * - ``pacman-smallclassic-vegetarian``
     - Paper smallClassic, Vegetarian Orange; DQN/PPO baselines and matching fixes/OFTEN.
   * - ``pacman-smallclassic-trapped``
     - Paper smallClassic, Trapped; DQN/PPO baselines and matching fixes/OFTEN.
   * - ``pacman-images-vegan``
     - Vegan with image observations; unconstrained image DQN only.

The ``pacman-smallclassic-*`` groups read the paper experiment IDs selected by
``train --paper``. The ``small`` map groups are separate exploratory configurations.
For example, to report the retained Vegan baseline data::

   python -m experiments.run report \
     --input experiments/results/pacman \
     --outdir experiments/output/smallclassic-vegan-report \
     --benchmark pacman-smallclassic-vegan --evaluation-name paper

All available methods are shown; missing methods are omitted. Different learner settings remain distinct
rows. Each row names its optimization targets; other norms remain monitored.
Tables use unscaled task return, environment-specific task metrics, episode
length, violations, separate training/teaching budgets and times, and fixing
latency where measured. Unknown timings are left blank, not replaced with zero.
Permission use and overlapping diagnostic counts are not added to violations.

Each training seed contributes one episode-mean value. Tables and bands show the
mean and sample standard deviation across those seeds. A single seed has zero
spread. Evaluation sample sizes are reported; independent baseline runs can have
different evaluation protocols. The base/fixes/OFTEN variants from the same run
use paired episode seeds. Identical Gardener bases shared by several teaching
variants are counted once per seed, with conflicting results rejected.

OFTEN curves restart at zero teaching transitions; base learning curves use
pretraining steps. Fixed pretrained policies appear as horizontal references,
not invented learning trajectories. The figures label the two step conventions;
the tables retain both budgets. Missing required norm measurements, incompatible
curve grids, different environment settings or evaluation episode limits, existing output
directories and overlapping input/output trees are
rejected. Reports include available completed seeds; check their counts before
using them as final benchmark results.

Local ``experiments/output/`` is ignored. Full paper-scale training is not a routine test or documentation smoke check.

LaTeX paper tables
~~~~~~~~~~~~~~~~~~

With the ``plots`` extra installed, generate the paper tables from the retained
results:

.. code-block:: console

   python -m experiments.generate_paper_tables \
     --output experiments/output/paper-tables

The output directory must be new and outside the results tree. The script
requires all selected recipes, seeds 0--7 and 1,000 episodes per seed. It writes
``pacman_baselines.tex``, ``grid_baselines.tex`` for Taxi, Merchant and Gardener,
and ``sources.json`` with exact evaluation paths and unrounded statistics.

Include the tables in your LaTeX document using ``\input{pacman_baselines.tex}``
and ``\input{grid_baselines.tex}``, with the ``booktabs`` and ``graphicx`` packages
loaded. Pacman spans the text width; the grid fits a single column.

Use ``--details`` to print source paths, ``--print-latex`` to print the tables,
or ``--table pacman`` / ``--table grid`` to select one table. ``--results``
selects a different root with the same paper dataset layout.
