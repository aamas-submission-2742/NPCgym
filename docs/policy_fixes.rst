Policy fixes
============

A *policy fix* selects an action using a planning model and the learned policy's
preferences. Planning can anticipate future norm violations instead of checking
only the immediate action. The policy's parameters and the task reward remain
unchanged. Use independent `monitors <monitors.rst>`__ to measure what actually
happens; a planner's predicted cost is not an observed violation count.

Install ``python -m pip install '.[asp]'`` from a checkout. This optional extra
supplies Clingo. Importing ``npc_gym`` or ``npc_gym.policy_fixes`` does not import
Clingo, Torch, or Stable-Baselines3. Constructing ``ASPPlanner`` without Clingo
raises an ``ImportError`` naming the required extra.

Built-in domain models are listed in their catalogues. The
`Pacman policy fixes <catalogue/pacman.rst#catalogue-pacman-policy-fixes>`__ cover
Vegan, both Vegetarian norms and Trapped, with runnable examples and explicit
movement, history and abstraction contracts.
`Gardener policy fixes <catalogue/gardener.rst#catalogue-gardener-policy-fixes>`__
cover collection, permissions, drainage and Rescue, including explicit history
updates and episode-ending obligations.
`Taxi policy fixes <catalogue/taxi.rst#catalogue-taxi-policy-fixes>`__ cover
reactive rain warnings;
`Merchant policy fixes <catalogue/merchant.rst#catalogue-merchant-policy-fixes>`__
cover Environment Friendly extraction attempts with an exact one-step model. For a complete learned-policy
comparison with paired evaluation and result files, run the
`Taxi example <examples.rst#taxi-policy-fixing>`__ or the masked
`Merchant example <examples.rst#merchant-policy-fixing>`__.

Essential contract
------------------

``ASPPlanner.solve(problem, action_values, *, proposed_action=None, allowed_actions=None)``
returns a ``PlanningDecision``. ``problem`` is a ``PlanningProblem(program, horizon=1)``;
``program`` is trusted ASP source describing the domain, admissible action
sequences and predicted costs. The horizon is the number of future actions,
starting at time zero. The returned ``plan`` has exactly this length. Execute
only its first action, then build a new problem from the observed transition.

``action_values`` maps integer action IDs to finite real Q-values. It must
include every modeled action. The default proposal is the highest-valued
currently allowed action, breaking ties by the lowest action ID.
``allowed_actions`` defaults to all keys and restricts the first action only;
future admissibility is specified by the domain model. An explicit proposal
must be allowed. It is promoted to preference rank zero; the remaining actions
retain their order by decreasing value, then increasing action ID. Thus a
caller can check exploratory proposals as well as greedy actions.

The planner accepts no environment reference and never executes an action.
Ordinary problems use a fresh solver; external-input problems can reuse one as
described below. Domain models must carry relevant normative history in their
snapshots or problem builders and clear it on reset. Only the caller advances
history after a real transition. Two independently live planners do not share
solver state.

Built-in models construct the planning problem for you. The
`problem interface <policy_fixes.rst#problem-interface>`__ and source-reuse
sections below are for authors of custom models.

Example: prefer a harmless alternative
--------------------------------------

This complete example has three actions. Action 0 is most attractive to the
policy but incurs a modeled violation. Actions 1 and 2 avoid it:

.. code-block:: python

   from npc_gym.policy_fixes import ASPPlanner, PlanningProblem

   problem = PlanningProblem('''
       candidate(0,0..2).
       cost("violations",1,harm) :- action(0,0).
   ''')
   decision = ASPPlanner().solve(problem, {0: 10.0, 1: 2.0, 2: 1.0})
   assert decision.proposed_action == 0
   assert decision.action == 1
   assert decision.changed and decision.optimal
   assert decision.costs == {"violations": 0, "policy": 1}

No environment step occurred. The cost of one under ``policy`` is the selected
action's rank, not a reward penalty.

Weights and priorities
----------------------

``ASPPlanner(*, objectives=None, conflict_limit=None, on_failure="raise", reuse_solver=True)`` uses
``{"violations": Objective(weight=1, priority=2), "policy": Objective(weight=1, priority=1)}``
by default. Configure every cost name used by the program. ``policy`` is required;
its weight can be zero to disable preference costs.

Each named cost is multiplied by its nonnegative integer weight. Larger positive
integer priorities dominate smaller ones, regardless of the lower-priority cost's
magnitude. Costs at the same priority are added. The solver minimizes these sums.
Negative costs can express bonuses. Equal optimal objectives are resolved by the
lexicographically smallest action sequence (first action ID, then second, and so
on). This tie rule applies below every configured objective priority.

For a weighted tradeoff rather than normative priority, continue the harmless
alternative example above:

.. code-block:: python

   from npc_gym.policy_fixes import Objective

   planner = ASPPlanner(objectives={
       "violations": Objective(weight=1, priority=1),
       "policy": Objective(weight=2, priority=1),
   })
   assert planner.solve(problem, {0: 10.0, 1: 2.0, 2: 1.0}).action == 0

Here violation costs one, whereas choosing action 1 would cost two in policy
deviation. Raw Q-value magnitudes never become optimization weights.

Results and failures
--------------------

``PlanningDecision`` contains ``proposed_action``, ``action``, the full ``plan``,
unweighted named ``costs``, weighted sums by priority in ``objective``, and
``status``. Cost mappings are detached and read-only. ``changed`` compares the
selected and proposed actions. ``optimal`` and ``expert_eligible`` are true only
when optimality has been proved; a model found before the search finishes is
not enough. Solver tie-breaking terms are excluded from ``objective``.

The plan is deterministic, but the model may admit several equally optimal
explanations for it. ``costs`` reports one such answer set: names sharing a
priority may have different individual costs while their weighted total stays
the same. Disabled objectives may also vary. This breakdown is unspecified and
can differ between fresh and reused solvers; equality of ``PlanningDecision``
instances compares it too. If you need reproducible per-name costs, make them
uniquely determined by the plan or give them distinct priorities and nonzero weights.

By default, ``NoPlanError`` reports unsatisfiability and ``PlanningLimitError``
reports a search stopped before a proof. ``conflict_limit`` is a nonnegative
integer bound on solver conflicts, not a wall-clock timeout or grounding limit.
``None`` leaves search unlimited. Malformed models raise ``PlanningModelError``;
invalid Python inputs raise ``TypeError`` or ``ValueError``.

With explicit ``on_failure="base_policy"``, unsatisfiability or an incomplete
solve returns the proposal with status ``unsatisfiable`` or ``limit``, an empty
plan, and empty predicted-cost mappings. Such a result is never expert evidence
or a valid basis for filtering replay. Programming errors and missing dependencies
still raise; they do not trigger fallback.

Policy integration
------------------

``FixedPolicy(planner, action_values, problem, *, allowed_actions=None)`` implements the callable contract
used by `evaluation <evaluation.rst>`__. Both callbacks receive
``(observation, info)``. The first returns the Q-value mapping; the second returns
a new ``PlanningProblem`` built from current public information. Encode current
action restrictions in the problem builder's time-zero candidates. Supply an
``allowed_actions(observation, info)`` callback when the base proposal itself
must respect an action mask; this prevents counting mask restrictions as
normative interventions. The adapter
retains ``last_decision`` for inspection and clears it before every call,
including calls that fail. It always replans; it does not execute a cached plan.

For example, an existing tabular learner can provide action values using
``{learner.action_start + i: float(q) for i, q in enumerate(learner.q_values(obs))}``.
This reads values without changing the table or consuming prediction randomness.
The domain problem builder remains independent of the learner implementation.

Teaching from fixes
-------------------

`OFTEN teaching <often.rst>`__ uses independent ordinary and demonstration streams
to teach tabular and DQN policies from these planning decisions. Saved policies run without
a planner.

Problem interface
-----------------

To write a custom model, return a ``PlanningProblem`` containing its ASP rules
and facts. A directly constructed problem is parsed and grounded on every solve.
The following sections describe its ASP contract and two optional ways to reuse
work between decisions.

The framework supplies these ASP predicates:

* ``time(T)`` for times zero through ``horizon - 1``.
* ``available_action(A)`` for each action-value key.
* ``preference(A,R)`` for the current policy ranking.
* ``action(T,A)``: exactly one candidate action is selected at each time.

The domain supplies ``candidate(T,A)``, constraints on selected actions, and
optional ``cost("name",Value,Key)`` atoms. Names are strings, values are signed
integers, and keys identify distinct contributions. For example,
``cost("violations",1,frog(0))`` and ``cost("violations",1,frog(1))`` contribute
two. Repeating the same atom does not duplicate it; giving the same name/key
two different values is an error. The framework adds the first-action rank as
``cost("policy",R,first_action)``. Other policy-cost keys may express domain-specific
future preference terms. No implicit future policy predictions are made.

Domain programs must not define the framework-supplied predicates or any
``npc_`` predicate, nor supply optimization directives or scripts. They are
trusted application code, not a format for executing untrusted input. Use the
provided cost interface for all optimization. Action IDs, times, weights,
priorities, and each weighted cost must fit the supported signed 32-bit numeric
range; Booleans are rejected as Python integer parameters.

The model owns transition dynamics, possible stochastic outcomes, deadline
history, termination/truncation, and any spatial window. If an episode ends
before the planning horizon, encode cost-free padding for the remaining actions.
The framework cannot infer whether a model captures all real outcomes. A
conservative model must account for adverse outcomes explicitly, rather than
let the solver select the most favorable future. Its guarantees remain limited
to its abstraction and planning horizon.

A domain adapter may use public full-state snapshots even when the learner sees
a smaller observation. Document this information advantage. Never obtain future
outcomes by advancing the real environment or reading future random draws.

Reusable domain source
----------------------

``PlanningProblem.from_parts(*, static, dynamic="", horizon=1)`` accepts two ASP
strings. ``static`` contains reusable rules and facts, such as action effects or
fixed map geometry. ``dynamic`` describes the current state. Static source is
cached automatically; both inputs may contain rules and facts, and either may be
empty. ``problem.program`` contains the static source, a newline, then the dynamic
source. Each input must contain complete ASP statements.

This complete example avoids whichever action is currently harmful:

.. code-block:: python

   from npc_gym.policy_fixes import ASPPlanner, PlanningProblem

   static_program = '''
       candidate(T,0..1) :- time(T).
       cost("violations",1,harm) :- action(0,A), harmful(A).
   '''
   planner = ASPPlanner()
   for harmful in (0, 1):
       problem = PlanningProblem.from_parts(
           static=static_program, dynamic=f"harmful({harmful}).", horizon=3,
       )
       assert planner.solve(problem, {0: 10.0, 1: 0.0}).action == 1 - harmful

Built-in and custom models can use the same bounded cache. Changing static text
requires no invalidation call. Each solve still grounds and validates the whole
program with a fresh solver. Construction requires no Clingo.

Static ``#include`` directives raise ``ValueError``; read files into strings first,
or include them in ``dynamic`` to reread them on every solve. Invalid input types
raise ``TypeError``; ASP syntax is checked when solving.

Reusable grounded programs
--------------------------

``PlanningProblem.from_externals(*, static, true_atoms=(), horizon=1)`` describes a
fixed model with changing Boolean inputs. ``static`` contains its rules, fixed
facts and ``#external`` declarations for every possible input. ``true_atoms``
lists the inputs that are true **on this decision**; all omitted externals are
false. Use canonical ground atoms without trailing periods, such as
``"occupied(north)"`` or ``"position(2,3)"``.

By default, the planner grounds the model once: it expands rules into concrete
instances, then updates the inputs between decisions. Reuse applies to built-in
and custom models alike. Pacman uses this form; Taxi, Merchant and Gardener use
``from_parts``.

This complete example grounds once and changes the harmful action:

.. code-block:: python

   from npc_gym.policy_fixes import ASPPlanner, PlanningProblem

   program = '''
       candidate(T,0..1) :- time(T).
       #external harmful(0..1).
       cost("violations",1,harm) :- action(0,A), harmful(A).
   '''
   planner = ASPPlanner()
   for harmful in (0, 1, 0):
       problem = PlanningProblem.from_externals(
           static=program, true_atoms=[f"harmful({harmful})"], horizon=3,
       )
       assert planner.solve(problem, {0: 10.0, 1: 0.0}).action == 1 - harmful

Each planner retains only its most recent prepared solver per thread. Changing
the source, horizon, action IDs, objectives or conflict limit rebuilds it. State
inputs, preference ranks and allowed actions are replaced on every solve;
there is no implicit episode history. Programs are validated when grounded,
inputs on each call, and decisions before returning. Independent planners and
threads do not share solvers.

Use ``ASPPlanner(reuse_solver=False)`` to create a fresh Clingo instance on every
decision. This rebuilds the external-input program with the same input validation
and objectives; parsed static source can still be cached. Ordinary problems and
those built with ``from_parts`` always use fresh solvers. Parsing caches are
independent of this option. With a conflict
limit, fresh and reused solvers may differ in whether they finish within it.

External declarations must use their default false value. Inputs must name
declared domain externals that survive grounding; unknown atoms, ordinary facts
and framework inputs raise ``PlanningModelError``. Static includes are forbidden.
Construction requires no Clingo. ``problem.program`` includes the true inputs as
facts. Copying this source into an ordinary ``PlanningProblem`` discards external
input validation; use ``reuse_solver=False`` to preserve the same contract.
Use ``from_parts`` when changing rules or an unbounded input domain makes full
pregrounding impractical; a larger grounded program is not necessarily faster.
