.. _catalogue-merchant:

Merchant
========

For planning support, see `Merchant policy fixes <#catalogue-merchant-policy-fixes>`__.

These monitors cover danger, deliveries, fighting, and tree extraction.
The initial observation counts as input 1. The recipe tables state what happens
after a match: ``prefix`` keeps the history; ``restart`` starts a fresh history
with the next input. Episode reset clears both history and counts.

.. list-table:: Find a monitor
   :header-rows: 1

   * - Monitor / type
     - ID
     - What it counts
   * - `Danger <#catalogue-merchant-simple>`__ (``SimpleMonitor``)
     - ``merchant/danger-v0``
     - Inputs spent in danger.
   * - `Delivery <#catalogue-merchant-simple>`__ (``SimpleMonitor``)
     - ``merchant/delivery-v0``
     - Sundown with an outstanding delivery.
   * - `Environment Friendly <#catalogue-merchant-simple>`__ (``SimpleMonitor``)
     - ``merchant/env-friendly-v0``
     - Extraction immediately after being at a tree with wood.
   * - `Pacifist <#catalogue-pacifist>`__ (``MultiMonitor``)
     - ``merchant/pacifist-v0``
     - Danger, plus fighting immediately after danger.
   * - `DeliveryPacifist <#catalogue-delivery-pacifist>`__ (``MultiMonitor``)
     - ``merchant/delivery-pacifist-v0``
     - Delivery failures, danger, and fighting immediately after danger.
   * - `Evolving <#catalogue-evolving>`__ (``ComplexMonitor``)
     - ``merchant/evolving-v0``
     - Extraction after a tree visit; from input 15 the visit must include wood.

.. _catalogue-merchant-simple:

Simple monitors
---------------

Each monitor exposes one ``count``. Propositions describe the current labels:
``danger`` means at danger, ``home`` at home, ``market`` at market, ``tree`` at a
tree, and ``wood`` carrying wood; ``sundown`` and ``extract`` have their literal
label meanings.

.. list-table:: Recipes
   :header-rows: 1
   :widths: 20 35 35 10

   * - Monitor
     - What increases its count
     - Regex
     - Policy
   * - Danger
     - Every input in danger.
     - ``.* [danger]``
     - prefix
   * - Delivery
     - Sundown while a delivery activated at home remains outstanding.
     - ``.*( [home & sundown] | [home & !market & !sundown] [!market & !sundown]* [sundown])``
     - prefix
   * - Environment Friendly
     - Extraction immediately after being at a tree while carrying wood.
     - ``.* [tree & wood] [extract]``
     - prefix

**Delivery:** Being at home activates the obligation, including at reset.
Reaching market fulfills it, but sundown on the same input takes precedence and
counts a failure. Completing it or missing the deadline ends the obligation.
Being at home on a later input can start it again. Ending the episode before
sundown does not count a failure.

**Environment Friendly:** Every extraction attempt immediately after an input
at a tree while carrying wood counts once, including unsuccessful attempts at
full capacity. Matching history is retained: if the tree remains intact, every
subsequent extraction attempt counts too. Collecting the first wood is allowed;
extracting ore or attempting extraction at an already depleted tree does not
violate this norm. Episode reset clears history and counts.

.. _catalogue-pacifist:

Pacifist
--------

This collection counts danger and a subsequent failure to avoid fighting.
The latter is named ``CTD`` for *contrary-to-duty*: a requirement that follows a
violation.

.. list-table:: Components
   :header-rows: 1
   :widths: 20 35 35 10

   * - Count key
     - What increases it
     - Regex
     - Policy
   * - ``Danger``
     - Every input in danger.
     - ``.* [danger]``
     - prefix
   * - ``CTD``
     - Fighting on the input immediately after danger.
     - ``.* [danger] [fight]``
     - prefix

The additional count ``Pacifist(total)`` sums the ``Danger`` and ``CTD`` counts. Both can increase
on the same input: danger followed by danger plus fighting increases the total
by two on the second input. Each match retains history, so repeated danger and
fighting can keep increasing both counts.

.. _catalogue-delivery-pacifist:

DeliveryPacifist
----------------

This norm base combines Delivery with both Pacifist components. Its independent
count keys are ``Delivery``, ``Danger`` and ``CTD``, with the meanings above.
The derived count ``DeliveryPacifist`` sums those three counts, counting each
event once. Simultaneous violations remain separate events.

The Merchant experiment penalties in ``experiments/merchant_bolts.py`` are
10,000 per Delivery failure, 100 per Danger violation and 1,000 per CTD (fighting)
violation. The adjustment is therefore
``-10000 * delivery_events - 100 * danger_events - 1000 * ctd_events`` on each step.
All three events together cost 11,100; the derived total adds no further penalty.
These weights express a stronger penalty for delivery failure, not a guarantee
of lexicographic priority. Environment Friendly is a separate norm base with
a penalty of 300 per violation. Values use raw task-reward units.

.. _catalogue-evolving:

Evolving
--------

This ``ComplexMonitor`` exposes one ``count`` for extraction immediately after
a qualifying tree visit. Visits on inputs **1–14** qualify without wood; visits
on input **15 onward** require wood. Reset is input 1. A visit on input 14 can
therefore cause an event on input 15 even without wood.

Its definition is a Boolean function of three ``SimpleMonitor`` instances:

.. code-block:: python

   evolving = (early & early_phase) | late

.. list-table:: Simple components
   :header-rows: 1
   :widths: 25 75

   * - Name
     - Meaning
   * - ``early``
     - Extraction immediately after a tree visit, without requiring wood.
   * - ``early_phase``
     - The current input is one of inputs 1–14.
   * - ``late``
     - Extraction after a qualifying visit from input 15 onward, including a pending visit carried over from input 14.

The components keep running throughout the episode. After a counted extraction,
that same input cannot also start the next event. The input number continues
increasing across events; episode reset starts the input numbering and event
count afresh.

.. _catalogue-merchant-policy-fixes:

Policy fixes
------------

``MerchantModel`` supplies one-step fixes for Environment Friendly.
It assigns a violation cost of one to Extract
when the agent is at an intact tree and already carries wood. All other actions
cost zero. With ``ASPPlanner()`` defaults, avoiding violations takes precedence;
among equally compliant actions, the learned policy's preference decides.

Abstraction and interface
~~~~~~~~~~~~~~~~~~~~~~~~~

Construct ``MerchantModel()`` and call ``model.problem(env.labeling_state())``
before each action. Pass all seven action values to ``ASPPlanner.solve``;
optionally restrict the choices with ``allowed_actions``. If Extract is the
only allowed action while prohibited, it is selected with predicted cost one.

Only the snapshot's current cell and whether carried wood is greater than zero
are needed. Position, layout, ore inventory, capacity, clock and danger risks
are irrelevant to this immediate violation. The model validates the two fields
it uses and does not inspect the others. It retains no history and can be reused
across resets, layouts and environments. Call only before episode end; the
snapshot does not contain a terminal flag.

The horizon is fixed at one. Costs exactly predict the next Environment Friendly
monitor event under the supplied Merchant labeler; the model does not predict
movement, later inventory changes, task reward or delivery success. Both facts
are already present in Merchant's ordinary observations, including the
``IgnoreTimeObservation`` view. No extra observation features are needed for
learning these fixes with OFTEN.

Example
~~~~~~~

This complete example uses fixed action preferences. Replace them with your
policy's Q-values. Install the optional ``asp`` extra to construct the planner.

.. code-block:: python

   from contextlib import closing

   from npc_gym.envs import MerchantEnv
   from npc_gym.monitors import MonitorInput, make_builtin_monitor
   from npc_gym.policy_fixes import ASPPlanner, MerchantModel

   model, planner = MerchantModel(), ASPPlanner()
   monitor = make_builtin_monitor("merchant/env-friendly-v0")
   with closing(MerchantEnv()) as env:
       observation, info = env.reset(seed=7)
       monitor.reset(MonitorInput(info["labels"]))
       values = {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0, 4: 2.0, 5: 0.0, 6: 0.0}
       for _ in range(20):
           decision = planner.solve(
               model.problem(env.labeling_state()), values,
               allowed_actions=[a for a, allowed in enumerate(info["action_mask"]) if allowed],
           )
           observation, reward, terminated, truncated, info = env.step(decision.action)
           previous_count = monitor.count
           monitor.update(MonitorInput(info["labels"], terminated, truncated))
           assert decision.costs["violations"] == monitor.count - previous_count
           if terminated or truncated:
               break
       print("Environment Friendly violations:", monitor.count)

For a learned-policy comparison, run the
`Merchant example <../examples.rst#merchant-policy-fixing>`__.
The `policy-fixes guide <../policy_fixes.rst>`__ explains objectives and optional
solver reuse. This model supports reuse through two Boolean external facts.

Manual review cases
~~~~~~~~~~~~~~~~~~~

.. list-table:: Immediate Extract costs
   :header-rows: 1

   * - Before the action
     - Predicted violations
   * - At an intact tree, carrying no wood (including carrying only ore).
     - Zero; the first wood is allowed.
   * - At an intact tree, carrying one or more wood resources.
     - One, regardless of whether extraction succeeds.
   * - At an intact tree with full inventory and some wood.
     - One on every attempt; the tree stays intact.
   * - At a rock while carrying wood.
     - Zero; ore extraction remains allowed.
   * - At a depleted tree after successful extraction.
     - Zero; the current cell no longer has the tree label.

The action mask is optional and remains separate from the norm: it normally
excludes extraction at full capacity. Even without that mask, every prohibited
attempt is counted. A compliant alternative can still be unproductive; task
performance depends on the learned policy's action values.
