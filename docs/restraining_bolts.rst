Restraining bolts
=================

A restraining bolt attaches a reward or penalty to a rule about the episode.
For example, it can subtract a reward whenever Pacman eats a ghost. It also adds
the rule's current automaton state to the observation, so the agent can learn
whether an earlier action has left a requirement pending.

Use a `monitor <monitors.rst>`__ when you want to count events. Use a bolt when
you also want the rule to affect learning. Both read the same
`labels <labeling_functions.rst>`__ and can use the same regex specifications.

Reward rule
-----------

A bolt consists of a temporal rule, a reward value, and a policy for retaining
history after a match. A **match** means that the processed label sequence
satisfies the rule. The wrapper evaluates every bolt independently on each step.

``make_regex_bolt(expression, *, propositions, reward, reporting="prefix", consume_initial=True)``
returns a ``BoltSpec``. It contains an immutable automaton definition, copied
proposition bindings, a finite non-Boolean real reward, and the two policy
options. Bindings have the contract ``MonitorInput -> bool`` and must be
stateless. ``make_ltlf_bolt`` takes a formula instead and uses the optional
compiler described in `Monitor specifications <monitor_specifications.rst>`__.

``RestrainingBoltWrapper(env, *, bolts={name: spec})`` runs each specification
with its own history. On each step it adds the reward of every matching bolt
to the wrapped reward. The returned reward equals the wrapped environment's reward plus the sum of the
configured rewards of all bolts that match on this step.

A bolt contributes its configured reward once per matching step. A negative
value gives a penalty; a positive value gives a bonus. Reset never pays a reward.

Example: penalizing danger
~~~~~~~~~~~~~~~~~~~~~~~~~~

Subtract one reward unit for every Merchant step spent in danger:

.. code-block:: python

   import gymnasium as gym
   import npc_gym
   from npc_gym.bolts import make_regex_bolt
   from npc_gym.envs.merchant.labels import MerchantLabel
   from npc_gym.wrappers import RestrainingBoltWrapper

   danger = make_regex_bolt(
       ".* [danger]",
       propositions={"danger": lambda step: MerchantLabel.AT_DANGER in step.labels},
       reward=-1,
   )
   env = RestrainingBoltWrapper(
       gym.make("npc_gym/Merchant-v2"), bolts={"danger": danger},
   )
   observation, info = env.reset(seed=0)
   observation, reward, terminated, truncated, info = env.step(0)
   details = info["restraining_bolts"]
   adjustment = details["reward_adjustments"]["danger"]
   assert reward == details["wrapped_reward"] + adjustment
   env.close()

The diagnostic gives the base reward and the separate adjustment, so the
assertion checks the reward equation directly. The regex uses the same
`syntax <monitor_specifications.rst>`__ as a simple monitor. A ``BoltSpec`` can be
reused across environments; each wrapper maintains independent execution state.

``make_regex_bolt``, ``make_ltlf_bolt`` and ``make_builtin_bolts`` minimize
DFAs by default, merging histories with identical future acceptance behavior
and removing unreachable states. This preserves reward events while reducing
the observation space. Pass ``minimize=False`` to preserve the compiler's
original state encoding. Training and evaluation must use the same setting:
minimization can change both state indices and one-hot dimensions. An existing
policy cannot generally be reused with a different encoding. Direct ``BoltSpec``
construction preserves its supplied definition; the wrapper does not minimize it.
See `compiled automata <monitor_specifications.rst#working-with-compiled-automata>`__
for compilation and minimization details.

Reusing a built-in rule
-----------------------

``make_builtin_bolts(monitor_id, *, reward, rewards=None, minimize=True)`` returns a read-only
mapping from ``<recipe-id>/<component-name>`` to ``BoltSpec``. For collections,
component names exactly match the member count keys in the `Monitor catalogue <monitor_catalogue.rst>`__,
including ``CTD`` for both Hungry Vegan Penalty and Penalty 3. For simple monitors,
the component name is the recipe name (for example, ``VegetarianBlue`` or ``Hungr``).
``npc_gym.monitors.builtins.builtin_regex_components(recipe_id)`` lists the exact
component names for either kind of recipe. Each regex component gets the default
``reward`` unless its full key appears in the ``rewards`` override mapping. Unknown keys and invalid rewards are rejected.
Derived counts are not additional reward events. The original recipe's reporting
and initial-input choices are preserved.

For example, assign different penalties to eating, missing a deadline, and
failing to pause:

.. code-block:: python

   from npc_gym.bolts import make_builtin_bolts

   recipe = "pacman/hungry-vegan-penalty-v1"
   bolts = make_builtin_bolts(
       recipe,
       reward=-1,
       rewards={f"{recipe}/Hungr": -5, f"{recipe}/CTD": -2},
   )

Here eating blue or orange costs one each, missing the hungry deadline costs
five, and failing to pause after eating costs two.

A collection's derived total does not create another penalty. Eating both
colors therefore produces two eating penalties, not a third penalty for the
total. This factory requires no optional compiler.

Evolving and Solution Guilt Maximum are not supported by this factory: their
Boolean combinations have no single automaton state to add to the observation.
Use ``MonitorWrapper`` to count their events. Existing mutable monitor instances
cannot be converted into bolts.

When rewards are applied
------------------------

Each matching step applies the bolt's reward, including consecutive matches
and a match on the final step. Several bolts can contribute on the same step.
They support two `reporting policies <monitors.rst#monitor-policies>`__:

* ``prefix`` keeps the history after a match.
* ``restart`` starts a fresh history for the next input, without reusing the
  matching input.

``episode_end`` reporting is available only for monitors. Reset clears history
and, by default, processes the initial labels. It can match and restart a rule,
but **reset never pays a reward**, either immediately or on a later step.
Use ``consume_initial=False`` to ignore the initial labels.

What the agent observes
-----------------------

The wrapper returns a dictionary with two entries:

.. code-block:: text

   {"observation": original_observation, "automata": features}

The original observation is preserved, even if it is itself a dictionary or
tuple. An **automaton state** summarizes the rule history needed to process the
next input. By default, ``features`` joins one vector per bolt, with a 1 at the
current state's position and 0 at every other position (a one-hot encoding). The vectors
follow the alphabetical order of bolt names. Their dtype is ``float32``, with a
matching Gymnasium ``Box`` space.

With ``state_encoding="index"``, each bolt instead contributes one integer,
using an ``int64`` vector and a ``MultiDiscrete`` space. Indices follow sorted
DFA state identifiers. An empty bolt collection gives an empty feature vector.
The state shown is always the one retained for the next step, including any
restart after a match.

``env.bolt_metadata[name]`` tells you the bolt's position (``index``), how its
DFA states map to indices (``state_indices``), and where its features appear
(``feature_slice``). The tabular learner accepts these dictionary observations
directly; neural agents can process the original observation and automaton
features separately. See the `Pacman image example <examples.rst>`__.

Using bolts alongside monitors
------------------------------

Use one ``RestrainingBoltWrapper`` per environment. The environment must provide
iterable ``info["labels"]``. Place labeling and time limits inside it, and
vector environments that reset automatically outside it. ``MonitorWrapper`` and
``RestrainingBoltWrapper`` can be in either order; each reads the same labels.
Evaluation does not infer monitor counts from bolt rewards or bolt diagnostics.

Reward scaling deserves care: a scaling wrapper **inside** the bolt wrapper
changes the base reward only. Placed **outside**, it also scales the bolt's
adjustments. The diagnostics report the values at the bolt wrapper itself.
Actions, seeds, options, action masks, and ending flags pass through unchanged.
After completion or a processing failure, reset before stepping again. An
already executed environment step cannot be undone.

Details for integrations
------------------------

``info["restraining_bolts"]`` contains the base ``wrapped_reward``, per-bolt
``reward_adjustments`` (including zeros), Boolean ``occurrences``, and retained
DFA ``states``. On reset the wrapped reward is ``None`` and all adjustments are zero.
Every observation and diagnostic dictionary is a fresh snapshot. The wrapper
requires the ``restraining_bolts`` key to be unused by the inner environment.

Two separately constructed ``BoltSpec`` objects compare as distinct even if they
express the same rule. Reuse a specification when you want to share its definition.

What the added memory guarantees
--------------------------------

Adding automaton memory does not by itself make every observation sufficient
for decision-making. A Markov-state guarantee also requires that the original
observation capture the environment dynamics and base reward, and that rule
conditions depend only on that state. Hidden labels, unobserved time limits, or
lossy images can break those assumptions. Likewise, assigning larger penalties
to some rules does not guarantee that an agent will always prioritize them.
