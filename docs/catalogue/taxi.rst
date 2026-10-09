.. _catalogue-taxi:

Taxi
====

For planning support, see `Taxi policy fixes <#catalogue-taxi-policy-fixes>`__.

Emergency checks warnings and safety during hurricanes. It ignores the reset
observation, so input 1 is the first environment step. Its simple components
all use ``prefix`` reporting: counting an event does not erase their history.

.. list-table:: Find a monitor
   :header-rows: 1

   * - Monitor / type
     - ID
     - What it counts
   * - `Emergency <#catalogue-emergency>`__ (``MultiMonitor``)
     - ``taxi/emergency-v0``
     - Missing rain-onset responses, missing safety deadlines, and unsafe hurricane stays.

.. _catalogue-emergency:

Emergency
---------

This collection counts four events separately. Safety means being at shelter,
or at home without flood risk, using the actual position after the action.
Arriving on a deadline satisfies safety; a blocked move cannot count as arrival.
A new hurricane starts a safety deadline:
**seven subsequent inputs if rain has never occurred before that input,
three if it has**. Rain on the hurricane input itself still selects seven.

.. list-table:: Components (all prefix)
   :header-rows: 1
   :widths: 20 40 40

   * - Count key
     - What increases it
     - Regex
   * - ``Warn Violations``
     - The input immediately after a rain onset has no warning.
     - ``( [rain] | .* [!rain] [rain] ) [!warn]``
   * - ``Stay Violations``
     - A hurricane input is unsafe, is not a new hurricane, and has no uncleared safety timer.
     - ``( [!new_hurricane]* | .* [safe] [!new_hurricane]*) [hurricane & !safe & !new_hurricane]``
   * - ``Seven-Step Safety Violations``
     - The seven-input safety deadline is reached while still unsafe.
     - ``[!rain]*( [new_hurricane & rain & !safe]N^7 | [new_hurricane & !rain & !safe](D^7 | Σ(j=0..6) D^j R N^(6-j)))``
   * - ``Three-Step Safety Violations``
     - The three-input safety deadline is reached while still unsafe.
     - ``.* [rain].* [new_hurricane & !safe] [!new_hurricane & !safe]^3``

Here ``N = [!safe]``, ``D = [!safe & !rain & !new_hurricane]``,
``R = [rain & !safe & !new_hurricane]``, and the sum notation Σ(j=0..6) joins the seven
alternatives with ``|``. ``rain``, ``warn``, ``new_hurricane``, and ``hurricane``
refer to the corresponding weather and action labels.

.. list-table:: Additional counts
   :header-rows: 1

   * - Count key
     - Calculation
   * - ``Safety Violations``
     - Sum of ``Seven-Step Safety Violations`` and ``Three-Step Safety Violations``.
   * - ``Emergency Violations``
     - Sum of the four component counts above.

A *rain onset* is a rainy input that is either the first processed input or follows
a dry input. Warn is required on the immediately following input, even if it is
dry. Warning on the onset itself does not satisfy this obligation. Persistent
rain requires no further warnings; each later dry-to-rain transition triggers a
new obligation. Reset labels are ignored. If the episode ends on the onset, no
missing-response violation is added; a final response input is checked normally.
For example, rain starting on input 4 requires Warn on input 5.

Reaching safety clears both timers, including on a deadline. A new hurricane
restarts its applicable timer; the two timers are independent and can expire on
the same input. Expiry counts once but leaves the timer uncleared, suppressing
Stay Violations until safety is reached.

For example, after earlier rain, a new hurricane on input 5 makes input 8 the
deadline. Safety on input 8 avoids the event; remaining unsafe counts one
``Three-Step Safety Violations`` event. Simply remaining unsafe on input 9 adds
no further safety-deadline event.

.. _catalogue-taxi-policy-fixes:

Policy fixes
------------

``TaxiModel`` in ``npc_gym.policy_fixes`` supplies one-step policy fixes for
**Warn only**, using Storm Taxi's seven actions. Safety deadlines and Stay remain
available as monitors but have no policy fixes. See the
`policy-fixes guide <../policy_fixes.rst>`__ for the optional ``asp`` extra and
solver configuration.

Contract
~~~~~~~~

Construct ``TaxiModel()`` after resetting the environment. After every actual
step, call ``advance(MonitorInput(labels, terminated, truncated))`` exactly once.
Do not pass reset labels. Call ``reset()`` after each subsequent environment reset;
use a separate model for each environment. Calls after an observed episode ending
are rejected until reset.

``problem()`` returns a one-step problem without changing history. Its only input
to ASP is ``warning_due``: whether the latest processed step started a rain spell.
When a warning is due, every action except Warn (6) costs one violation; otherwise
all seven actions cost zero. No positions, passengers, weather predictions or
safety timers are modeled. The static rules and external fact support solver reuse.

``ASPPlanner()`` first minimizes warning violations, then policy preference rank.
With Warn available, this prevents all Warn violations. If ``allowed_actions``
excludes Warn when it is due, every permitted action costs one violation. Plain
Taxi has no Warn action and is outside this model's scope.

Example
~~~~~~~

This complete example uses fixed action preferences. Replace them with the
learner's Q-values. The Emergency monitor reports all components, while the
planner minimizes only Warn.

.. code-block:: python

   from gymnasium.wrappers import TimeLimit

   from npc_gym.envs import StormTaxiEnv
   from npc_gym.monitors import MonitorInput
   from npc_gym.monitors.taxi_monitors import EMERGENCY_NORM_ID, make_taxi_monitor
   from npc_gym.policy_fixes import ASPPlanner, TaxiModel

   env = TimeLimit(StormTaxiEnv(), max_episode_steps=20)
   try:
       _, info = env.reset(seed=7)
       model, planner = TaxiModel(), ASPPlanner()
       monitor = make_taxi_monitor(EMERGENCY_NORM_ID)
       monitor.reset(MonitorInput(info["labels"]))
       while True:
           decision = planner.solve(model.problem(), dict.fromkeys(range(7), 0.0))
           _, _, terminated, truncated, info = env.step(decision.action)
           actual = MonitorInput(info["labels"], terminated, truncated)
           model.advance(actual)
           monitor.update(actual)
           if terminated or truncated:
               break
       assert monitor.counts["Warn Violations"] == 0
   finally:
       env.close()

Manual review cases
~~~~~~~~~~~~~~~~~~~

.. list-table:: Warning responses
   :header-rows: 1

   * - History
     - Next action
   * - Reset, including a rainy reset observation.
     - Follow policy preferences; reset creates no obligation.
   * - First processed input is rainy, or a dry input is followed by rain.
     - Warn, even if the onset action was already Warn.
   * - Response step has passed; rain persists.
     - Follow policy preferences, whether the response was satisfied or missed.
   * - Rain stops and starts again.
     - Warn immediately after the new onset.
   * - Rain begins on the final episode step.
     - No further action and no extra missing-response event.

.. _taxi-policy-fixes-often:

Learning with OFTEN
~~~~~~~~~~~~~~~~~~~

The fixer remembers rain history. For OFTEN to learn the same behavior, the
learner's observations must also distinguish a rain onset from continuing rain,
for example through an added warning-due feature or observation history. Current
rain status alone does not provide that distinction.
