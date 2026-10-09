.. _catalogue-gardener:

Gardener
========

For planning support, see `Gardener policy fixes <#catalogue-gardener-policy-fixes>`__.

Collect One checks whether at least one frog was collected during the episode.
The reset observation is ignored; input 1 is the first environment step. Its
``prefix`` policy keeps the history until episode reset.

.. list-table:: Find a monitor
   :header-rows: 1

   * - Monitor / type
     - ID
     - What it counts
   * - `Collect One <#catalogue-collect-one>`__ (``SimpleMonitor``)
     - ``gardener/collect-one-v0``
     - Ending the episode without collecting a frog.

.. _catalogue-collect-one:

Collect One
~~~~~~~~~~~

.. list-table:: Simple monitor; exposes one ``count``
   :header-rows: 1
   :widths: 20 35 35 10

   * - Monitor
     - What increases its count
     - Regex
     - Policy
   * - Collect One
     - An episode-ending input with no collection on any processed input.
     - ``[!collected]* [!collected & end]``
     - prefix

``collected`` means at least one ``FrogCollected`` label; ``end`` means terminated
or truncated. Collection on the final input fulfills the requirement, so it
counts no failure. Collecting several frogs still fulfills just this one
requirement. A collection label on reset is ignored.

Object events and rescue
------------------------

``make_gardener_monitor(id, *, num_frogs=None, num_puddles=None, minimize=True)`` in
``npc_gym.monitors.gardener_monitors`` constructs the following additional
recipes. Supply the environment's ``num_frogs`` for collection and per-frog Rescue recipes and
``num_puddles`` for Drain. These sizes keep count keys fixed across episodes.
All recipes ignore reset labels and require no optional dependency.

* ``gardener/no-collect-v0`` counts every collected frog, with members
  ``NoCollect/<frog-id>`` and sum ``NoCollect``.
* ``gardener/drain-v0`` counts each ``PuddleDrained`` event, with members
  ``Drain/<puddle-id>`` and sum ``Drain``. A drainage label means that at least
  one uncollected frog was nearby; empty drainages are not violations.
* ``gardener/collect-permission-v0`` counts every collected frog when
  ``PERMITTED_COLLECT`` occurs (collection while moving right), with members
  ``CollectPerm/<frog-id>`` and sum ``CollectPerm``. This records permission use.
* ``gardener/permission-aware-v0`` records ``Collected/<frog-id>``,
  ``Permitted/<frog-id>``, and ``Unpermitted/<frog-id>`` independently, with
  corresponding sums ``Collected``, ``Permitted``, and ``Unpermitted``.
  Each unpermitted event is the Boolean composition **collected AND NOT
  permitted**. Unlike unconditional No Collect, permitted collection is exempt.

For example, collecting two frogs while moving right increases Collected and
Permitted by two each and Unpermitted by zero. A later unpermitted collection
increases Collected and Unpermitted by one each. No implicit total combines
these overlapping measurements.

``gardener/rescue-v1`` is a stateful ``SimpleMonitor``. Each drainage creates
one obligation per nearby frog. Collection on the triggering input satisfies
it immediately. Otherwise, collection through the fifth subsequent input is
in time. On the sixth subsequent input, expiry wins over collection. Repeated
drainages create separate obligations, and collecting a frog fulfills all its
unexpired obligations. Termination or truncation fails all obligations still
pending after that input's collections and new activations.

This version counts **inputs with at least one failed obligation**, at most once
per input, regardless of how many frogs have failed obligations.

``gardener/rescue-v2`` uses the same activation, deadline and collection rules,
but counts failures **separately per frog**. It returns a ``MultiMonitor`` with
members ``Rescue/<frog-id>`` and their sum ``Rescue``. Supply ``num_frogs``.
Two frogs failing together add two to the total. Several obligations for the
same frog failing together add only one; failures on different inputs each
count. A collection fulfills only that frog's unexpired obligations.

Each member uses a minimized DFA by default. ``minimize=False`` retains compiler
states without changing counts. ``rescue_recipe(frog_id)`` returns its
``RegexRecipe`` and stateless proposition bindings for constructing an equivalent
bolt. Both versions ignore reset labels and clear history on reset.

For example, drainage on step 1 permits collection through step 6. If still
pending on step 7 it fails, even if collection occurs on step 7. Episode ending
on step 4 fails an outstanding obligation early. Reset clears all history.

.. _catalogue-gardener-policy-fixes:

Policy fixes
------------

``GardenerModel`` supports **No Collect**, **Drain**, and **permission-aware
collection**. It searches short local action plans and considers all legal frog
moves, without estimating their probabilities. Execute the first action and
replan from the new snapshot. The temporal Collect One and Rescue monitors remain
available for evaluation, but are not supported by this model. Collect Permission
records permission use and is not a violation objective.

Use ``norm_id="gardener/permission-drain-v0"`` to combine permission-aware
collection with Drain. Its violation cost is the sum of unpermitted collections
and harmful drainages, each with weight 5 under the default objectives. Legal
Right collections remain exempt; permission does not exempt drainage. For
example, an unpermitted collection and one harmful drainage cost two violations;
a permitted collection with that same drainage costs one. Evaluate this recipe
with the permission-aware collection and Drain monitors described above;
``Collected`` and ``Permitted`` are descriptive measurements, not extra penalties.

``GardenerModel(initial_state, *, norm_id=NO_COLLECT_NORM_ID, horizon=1,
radius=4, puddle_respawn=20, frog_freeze=5)`` accepts the immutable snapshot from
``env.labeling_state()``. Supply the environment's actual refill and freeze
parameters. ``problem(state, *, remaining_steps=None)`` returns a
``PlanningProblem``; a positive remaining budget shortens the horizon if needed.
Supply values for all five actions. See the `policy-fixes guide
<../policy_fixes.rst>`__ for installation, objectives and solver errors.

The model retains no episode history and needs no advancement callback. It can
be constructed mid-episode and reused after reset while layout and frog identities
match. It validates snapshots but cannot detect episode end: the snapshot has no
terminal flag. Plan only before termination or truncation. Neither planning nor
encoding changes environment state or RNG.

Example
~~~~~~~

This complete example fixes a fixed preference order for No Collect. Replace
``values`` with a trained learner's action values.

.. code-block:: python

   from contextlib import closing

   from npc_gym.envs import GardenerEnv
   from npc_gym.policy_fixes import ASPPlanner, GardenerModel

   with closing(GardenerEnv(size=5)) as env:
       env.reset(seed=7)
       model = GardenerModel(env.labeling_state())
       planner = ASPPlanner(objectives=model.objectives)
       values = {0: 4.0, 1: 3.0, 2: 2.0, 3: 1.0, 4: 0.0}
       for step in range(5):
           problem = model.problem(env.labeling_state(), remaining_steps=5 - step)
           decision = planner.solve(problem, values)
           _, _, terminated, truncated, _ = env.step(decision.action)
           if terminated or truncated:
               break

Local transition model
~~~~~~~~~~~~~~~~~~~~~~

The window is a square centered on the current agent, including its boundary.
The default radius four gives a 9 × 9 view. Frogs and puddles initially outside
it are ignored; frog paths leaving it are discarded. The agent must stay inside
it. The window is rebuilt for each decision. Choosing a radius at least as large
as the horizon avoids restricting otherwise legal agent routes.

1. The agent proposes Right (0), Up (1), Left (2), Down (3) or Stay (4).
   Walls, puddles and grid boundaries turn blocked proposals into Stay.
   Blocked Right therefore does not authorize collection.
2. Each mobile frog moves one legal cardinal step. All such moves have positive
   support in the environment, so all are considered. A frog stays only when
   frozen or when no cardinal move is legal in the actual grid. The observation
   window itself does not create an artificial Stay option.
3. Frogs sharing the agent's resulting cell are collected. Those branches are
   removed before drainage and cannot cause later collections or drainages.
   Surviving frogs' positive freeze timers decrease by one.
4. Puddle timers decrease; reaching zero refills a puddle before drainage.
   Cardinal adjacency drains a full puddle and restarts its refill timer.
   Uncollected frogs in its eight-cell neighborhood are frozen. A harmful drainage
   occurs only when such a frog remains nearby **after collection**.

Grass, task rewards, score-limit termination and future replanning are not
simulated. The task policy supplies action preferences. Predictions beyond an
unmodeled episode ending can therefore be pessimistic; local clipping can miss
incoming frogs. These are abstraction limits, not global safety guarantees.

Costs
~~~~~

Use ``ASPPlanner(objectives=model.objectives)`` for equal-priority costs:
``violations`` has weight 5 and ``policy`` has weight 1. The policy cost is the
first action's preference rank, from zero through four. A plain ``ASPPlanner()``
uses the general norm-first priorities instead.

Select ``norm_id`` using a constant from ``npc_gym.monitors.gardener_monitors``:

.. list-table:: Unweighted predicted violation costs
   :header-rows: 1

   * - Norm
     - Count minimized
   * - ``NO_COLLECT_NORM_ID``
     - One per frog that could be collected anywhere in the plan.
   * - ``PERMISSION_AWARE_NORM_ID``
     - One per frog that could be collected on an action other than effective Right.
   * - ``DRAIN_NORM_ID``
     - One per puddle and step that could drain with an uncollected frog nearby.

Alternative collection times never add multiple costs for the same frog.
A permitted collection removes that frog on that path; it cannot later become
an unpermitted collection. Drainage still counts separate events, even if the
same frog is nearby each time. Possible drainage events from incompatible frog
paths may be combined, so their sum need not be achievable in a single run.
These costs are neither expectations nor measured monitor counts.

For manual review: two frozen frogs at (2, 1), with the agent at (1, 1), are
both collected by Right: No Collect costs two and unpermitted collection costs
zero. A blocked Right instead executes Stay and grants no permission. A single
frog collectible on either of two future steps contributes one collection cost.
A frog collected before a nearby puddle drains contributes no drainage violation.

Relationship to the reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The released `Gardener ASP model
<https://gitlab.tuwien.ac.at/martin.tappler/OFTEN-DeepRL/-/blob/3546dfd846f98e7f542b6e6107454bd5e2a5539c/gardener_program.lp>`__
uses the same central abstraction: a local square, short fixed action plans,
and adverse cardinal frog moves. NPC Gym's default of one future action and radius
four matches its Gardener experiment's horizon two (which includes the initial
state). This is an adaptation to NPC Gym, not an exact reproduction of its game
or objective.

.. list-table:: Shared abstraction and necessary differences
   :header-rows: 1

   * - Aspect
     - Released reference
     - NPC Gym
   * - Task
     - Reach a target; avoid destroying plants and frogs.
     - Mow and drain; selected norms concern collection and harmful drainage.
   * - Agent movement
     - Four cardinal actions; blocked plans are infeasible.
     - Five actions; blocked proposals execute as Stay.
   * - Frog dynamics
     - Cardinal movement; no freeze timers or explicit removal in predicted paths.
     - Cardinal movement with freeze timers; collected paths disappear.
   * - Violation accounting
     - Time-weighted frog encounter penalties; simultaneous encounters share a penalty.
     - Each collection counted once per frog; drainage once per puddle and step.
   * - Policy preference
     - Ranks at hypothetical positions, time weighting and visit-dependent multiplier.
     - Current first-action rank; equal-priority norm weight 5 and policy weight 1.
   * - Task-specific terms
     - Target bonus and repeated-cell penalty.
     - Neither; these terms are not part of NPC Gym's maintenance task.

The reference uses saturation to check adverse frog movements. This model
propagates reachable states per frog instead; it does not enumerate joint frog
trajectories. Both approaches ask whether a harmful encounter is possible,
but the event accounting above deliberately differs.

Observations for OFTEN
~~~~~~~~~~~~~~~~~~~~~~

``StateFeatureObsWrapper(env, include_frogs=True)`` preserves the twelve compact
task features and appends fifteen values, giving a 27-element ``float32`` vector.
All values lie in [0, 1]. The option defaults to ``False``, which retains the
twelve task features alone. Import the wrapper from
``npc_gym.wrappers.gardener_wrappers``.

.. list-table:: Additional features, using zero-based indices
   :header-rows: 1

   * - Indices
     - Meaning
   * - 12–16
     - Possible collections, divided by the total frog count.
   * - 17–21
     - Possible harmful drainages, divided by the total puddle count.
   * - 22–26
     - Action legality: one if the proposal can execute, otherwise zero.

Each group is ordered **Right, Up, Left, Down, Stay**. A zero denominator gives
zero. Collected frogs contribute no risk, but remain in the total used for
normalization. Blocked proposals have Stay's risks and legality zero; Stay itself
is legal. Thus a blocked Right can be distinguished from a permitted collection
while moving Right.

The ten risk features consider all legal frog moves, frozen frogs staying in
place, and puddles refilling on the current step. Collection precedes drainage.
They are computed directly from the supplied observation in Python, without
Clingo, environment mutation or random draws.

Collection counts are the maximum possible number collected on that step.
Drainage counts include each puddle that could have an uncollected frog nearby;
the possibilities need not occur together. For example, a single frog able to
move near either of two puddles can produce a count of two although at most one
harmful drainage can occur. These are possible-event counts, not probabilities.
They match the default one-step planner's costs before normalization; for
permission-aware collection, a legal Right has zero unpermitted cost.

.. code-block:: python

   from contextlib import closing

   from npc_gym.envs import GardenerEnv
   from npc_gym.wrappers.gardener_wrappers import StateFeatureObsWrapper

   with closing(StateFeatureObsWrapper(GardenerEnv(), include_frogs=True)) as env:
       observation, _ = env.reset(seed=7)
       assert observation.shape == (27,)

Use this same encoding for pretraining, teaching and evaluation. These summaries
expose the default fixer's immediate risks, but do not determine longer-horizon
plans. A planner using a smaller local radius may also omit events included in
the features. See the `OFTEN observation contract
<../often.rst#learner-observations>`__.

``LocalGridObsWrapper(env, radius=4, include_frogs=True)`` is an alternative
spatial representation. It adds a channel for each freeze timer, counting
uncollected frogs per cell divided by the total frog count. Collected and
out-of-window frogs are omitted. Both wrappers omit frogs by default.
