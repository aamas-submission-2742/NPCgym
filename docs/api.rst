Public API
==========

This reference locates the public classes and functions under ``npc_gym``.
Import domain-specific labels from ``npc_gym.envs.<domain>.labels`` and monitor
factories from ``npc_gym.monitors.<domain>_monitors``. The guides linked below
explain their behavior; the tables identify where to import them.

See `Monitors <monitors.rst>`__ for the ``SimpleMonitor``, ``ComplexMonitor``, and
``MultiMonitor`` interfaces, reset behavior, and composition rules.

Environment and label interfaces
--------------------------------

.. list-table:: Imports for environments and labels
   :header-rows: 1

   * - Module
     - Public names
   * - ``npc_gym.envs``
     - ``StormTaxiEnv``, ``MerchantEnv``, ``GardenerEnv``, ``PacmanEnv`` and immutable state snapshots.
       Pacman also exports ``PacmanLayout``, ``GhostConfig`` and ``PacmanSnapshot``.
   * - ``npc_gym.labels``
     - ``Label``, ``LabelSet``, ``Transition``, ``LabelingFunction`` and their state/action type variables.

Rules and wrappers
------------------

.. list-table:: Imports for monitoring and reward rules
   :header-rows: 1

   * - Module
     - Public names
   * - ``npc_gym.automata``
     - ``CompiledDFA``: reusable deterministic automata for rules over Boolean conditions.
   * - ``npc_gym.monitors``
     - ``Monitor`` (a type union), ``SimpleMonitor``, ``ComplexMonitor``, ``MultiMonitor``, ``MonitorInput``, ``MonitorFactory``, ``MonitorRegistry`` and registry exceptions;
       ``make_builtin_monitor``, ``AutomatonMonitor``, ``Reporting`` and ``Proposition`` for compiled finite-trace monitoring.
   * - ``npc_gym.monitors.collection``
     - Functions to create, reset, update, and collect counts from groups of monitors.
   * - ``npc_gym.monitors.builtins``
     - ``builtin_regex_components`` returns immutable mappings of catalogue component names to ``RegexRecipe`` specifications.
       ``RegexRecipe.compile()`` shares a compiled DFA; ``make(environment)`` creates a fresh monitor.
   * - ``npc_gym.monitors.ltlf``
     - ``from_ltlf``, ``LTLfCompiler``, ``MonaCompiler``, ``LTLfCompilationError``.
       Default compilation requires the ``ltlf`` extra and MONA.
   * - ``npc_gym.monitors.regex``
     - ``compile_regex``, ``from_regex``, ``RegexLimits``, ``RegexSyntaxError`` and ``RegexCompilationError``.
       Boolean trace-regex compilation uses only core dependencies.
   * - ``npc_gym.bolts``
     - ``BoltSpec``, ``make_regex_bolt``, ``make_ltlf_bolt``, ``make_builtin_bolts``: immutable reward and automaton specifications.
   * - ``npc_gym.wrappers``
     - ``LabelingWrapper``, ``LabelMode``, ``StateExtractor``, ``MonitorWrapper``, ``RestrainingBoltWrapper``, ``BoltMetadata``.

Learning and evaluation
-----------------------

.. list-table:: Imports for learning and recording results
   :header-rows: 1

   * - Module
     - Public names
   * - ``npc_gym.algorithms``
     - ``TabularQLearning``, ``TrainingCallback``, ``TrainingStep``.
       ``TabularOFTEN``, ``TeachingSession``, ``TeachingStats``, ``TeachingStep`` support
       `OFTEN teaching <often.rst>`__.
   * - ``npc_gym.policy_fixes``
     - ``ASPPlanner``, ``PlanningProblem``, ``Objective``, ``PlanningDecision``, ``FixedPolicy``,
       ``ActionValuePolicy``, ``PlanningError``, ``NoPlanError``, ``PlanningLimitError``, ``PlanningModelError``.
       See `Policy fixes <policy_fixes.rst>`__; constructing a planner requires the ``asp`` extra.
       ``PacmanModel`` supplies snapshot-based planning for Vegan and both Vegetarian norms;
       see `Pacman policy fixes <catalogue/pacman.rst#catalogue-pacman-policy-fixes>`__.
       ``GardenerModel`` plans local collection and drainage fixes without normative history;
       see `Gardener policy fixes <catalogue/gardener.rst#catalogue-gardener-policy-fixes>`__.
       ``TaxiModel`` and ``MerchantModel`` supply Warn and Environment Friendly planning;
       see `Taxi policy fixes <catalogue/taxi.rst#catalogue-taxi-policy-fixes>`__ and
       `Merchant policy fixes <catalogue/merchant.rst#catalogue-merchant-policy-fixes>`__.
   * - ``npc_gym.evaluation``
     - ``Policy``, ``evaluate``, ``EpisodeResult``, ``EvaluationSummary``, ``TerminationClass``,
       ``EpisodeCSVWriter``, ``LearningCurveCSVWriter``, ``EvaluationJSONWriter``, ``EPISODE_METRICS_KEY``.
       ``plot_learning_curve`` saves figures from standalone curve CSVs; calling it requires the ``plots`` extra.
   * - ``npc_gym.integrations.sb3``
     - ``SB3Policy``, ``SB3EvaluationCallback``, ``IntermediateEvaluation``; requires the ``sb3`` extra.
       ``DQNOFTEN`` adds `DQN OFTEN teaching <often.rst#dqn-teaching>`__ with ``asp`` planners.
   * - ``npc_gym.integrations.optuna``
     - ``OptunaReporter``; requires the ``optuna`` extra at construction.

Environment labels
------------------

``npc_gym.envs.taxi.labels`` exports ``TaxiActionLabel``, ``TaxiWeatherLabel``, ``TaxiLocationLabel``,
``TaxiPassengerLabel``, ``TaxiLabelingFunction``, ``TaxiAuthorityState``, and ``TaxiState``.

``npc_gym.envs.merchant.labels`` exports ``MerchantLabel``, ``MerchantLabelingFunction``, ``MerchantAuthorityState``,
and ``MerchantState``. ``npc_gym.envs.gardener.labels`` exports ``GardenerLabel``, ``GardenerLabelingFunction``,
``GardenerState``, ``FrogCollected``, and ``PuddleDrained``.

``npc_gym.envs.pacman.labels`` exports ``PacmanLabel``, ``PacmanLabelingFunction``, ``PacmanLabelTracker``,
``PacmanAuthorityState``, ``PacmanLayoutState``, ``PacmanPlayerState``, ``PacmanGhostState``, and ``PacmanState``.

Registered monitors
-------------------

Each ``npc_gym.monitors.<domain>_monitors`` module exports its norm-ID constants, public registry, and
``make_<domain>_monitor`` factory. See the `monitor catalogue <monitor_catalogue.rst#monitor-catalogue>`__
for identifiers and behavior. A custom monitor can be passed directly to a
wrapper or evaluator through a no-argument factory; registration is optional.

Domain wrappers
---------------

Task-specific wrappers are intentionally explicit: ``IgnoreWeatherRelevant`` in ``taxi_wrappers``;
``IgnoreTimeObservation`` in ``merchant_wrappers``; and ``StateFeatureObsWrapper``, ``LocalGridObsWrapper``, and
``IllegalActionPenaltyWrapper`` in ``gardener_wrappers``. They are not re-exported from ``npc_gym.wrappers`` because
their contracts apply only to the corresponding domain.

Tabular learning and persistence
--------------------------------

``TabularQLearning`` requires a discrete action space and observations it can
encode as table keys. ``learn(total_timesteps)`` takes a cumulative target:
after learning 100 transitions, calling ``learn(150)`` requests 50 more.
``predict`` can select actions deterministically or stochastically.
``save`` and ``load`` use a versioned ZIP format without pickle.

Set ``use_action_mask=True`` only when every reset and step supplies a valid
mask. Use a separate environment for evaluation so it cannot change the state
of the training environment. See the `standalone examples <examples.rst>`__
for complete learning, saving, loading, and evaluation workflows.
