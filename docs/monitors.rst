Monitors
========

A monitor counts events during an episode. For example, it can count how often
Pacman eats a blue ghost, or how often a delivery remains unfinished at sundown.
It reads the `labels <labeling_functions.rst>`__ attached to each step and keeps
whatever history its rule needs.

Monitors do not change the environment's rewards or observations. To attach a
reward or penalty to a temporal rule, use a `restraining bolt <restraining_bolts.rst>`__.
To choose an existing monitor, see the separate `Monitor catalogue <monitor_catalogue.rst>`__.

Inputs, occurrences, and counts
-------------------------------

An **input** describes one reset or environment step. It is represented by
``MonitorInput(labels, terminated=False, truncated=False)``: a frozen set of labels
and two flags indicating task termination and external truncation. A **trace** is
a finite sequence of these inputs, in processing order.

An **occurrence** is a Boolean answer to “does this event occur on this input?”
The answer may depend on earlier inputs. An **event count** is the number of processed
inputs on which that answer was true since the latest episode reset. Several
ways of satisfying one event on the same input still increase its count by only one.

There are three monitor types:

.. list-table:: Monitor types
   :header-rows: 1

   * - Type
     - Definition
     - Result of ``update(input)``
   * - ``SimpleMonitor``
     - Detects one event, keeping any history that its rule requires.
     - One Boolean occurrence; updates its ``count``.
   * - ``ComplexMonitor``
     - A simple monitor or a Boolean combination of other event monitors.
     - One Boolean occurrence; updates its ``count``.
   * - ``MultiMonitor``
     - A collection of named event monitors and optional derived counts.
     - A mapping of member names to occurrences; exposes named ``counts``.

A **derived count** is calculated from member counts, for example their sum. It
is not accumulated separately. A collection has no single occurrence or implicit
total. Reset may itself count
as an input: ``consume_initial`` specifies that choice for each event monitor.

For environment integration, go to `Using monitors with an environment <#monitor-wrapper>`__. For reset behavior
and the rules for sharing monitors, see `Reset and ownership <#monitor-contracts>`__.

A simple monitor
----------------

A ``SimpleMonitor`` exposes ``update(input) -> bool`` and an integer ``count``.
The count starts at zero. Each successful update increases it by one if the
returned occurrence is true and leaves it unchanged otherwise. Custom subclasses
implement ``detect(input) -> bool``; the monitor handles the counting.

``from_regex(expression, propositions=..., reporting="prefix", consume_initial=True)``
constructs an automaton-based ``SimpleMonitor``. Each proposition name maps to a
function ``MonitorInput -> bool``. These functions must be stateless; the
automaton maintains the rule's history. The reporting policy decides when a
match counts, as described `below <#monitor-policies>`__.

For example, if labels contain ``"blue"`` whenever a blue ghost is eaten,
``.* [blue]`` counts each such input:

.. code-block:: python

   from npc_gym.monitors import MonitorInput
   from npc_gym.monitors.regex import from_regex

   def eating(color):
       return from_regex(
           f".* [{color}]",
           propositions={color: lambda step: color in step.labels},
       )

   blue = eating("blue")
   blue.reset(MonitorInput(frozenset()))
   assert blue.update(MonitorInput(frozenset({"blue"}))) is True
   assert blue.update(MonitorInput(frozenset())) is False
   assert blue.count == 1

``[blue]`` means “blue is true on this input.” The preceding ``.*`` allows any
history before it. The ``propositions`` dictionary tells the monitor how to
read ``blue`` from the labels. Here the first update returns ``True`` and adds one;
the next returns ``False`` and leaves the count at one.

Combining events
----------------

The operators ``&``, ``|``, and ``~`` construct a ``ComplexMonitor`` from other
event monitors. They combine the components' Boolean events on the **same
input**, rather than their accumulated counts:

.. list-table:: Boolean event rules
   :header-rows: 1
   :widths: 30 70

   * - Expression
     - Returns ``True`` when
   * - ``a & b``
     - Both ``a`` and ``b`` register an event on this input.
   * - ``a | b``
     - At least one of ``a`` or ``b`` registers an event on this input.
   * - ``~a``
     - ``a`` does not register an event on this input.

The result has the same ``update(input) -> bool`` and ``count`` contract as a
``SimpleMonitor``. Operands must agree on ``consume_initial``. A ``SimpleMonitor`` is a
subclass of ``ComplexMonitor``, so combinations can be nested:
``(first | second) & ~exception``.

Using the ``eating`` factory and imports from the preceding example, count an
input where either color is eaten:

.. code-block:: python

   either_color = eating("blue") | eating("orange")
   either_color.reset(MonitorInput(frozenset()))
   both_colors = MonitorInput(frozenset({"blue", "orange"}))
   assert either_color.update(both_colors)
   assert either_color.count == 1

Both colors were eaten, but the combined question—“was either color eaten?”—has
one yes-or-no answer. The count increases once.

Similarly, ``violation & ~exception`` counts a violation only when the exception
does not occur on that same input. ``~exception`` means “no exception now,” not “there
has never been an exception.”

A combination keeps its component monitors running in the background without
building a new automaton. Every distinct component instance is updated exactly
once per input, even if its result cannot change the combined answer. This also
applies when an instance appears in several expressions. Its history and reset
policies continue to apply. An exception can suppress the combined event; it does not
cancel a pending obligation inside another component.

Keeping separate counts
-----------------------

``MultiMonitor(members, *, derived=None)`` takes a mapping of names to
``ComplexMonitor`` instances. ``update(input)`` advances the members, updating shared
components once, and returns a read-only mapping ``{name: occurred}``.
``counts`` returns a read-only snapshot containing each member's current count
and any derived values. ``members`` provides read-only access to the named
monitor instances.

Each derived function receives a read-only snapshot of **member counts only**.
It must be stateless and return a nonnegative integer, excluding Booleans.
Derived values are calculated when ``counts`` is read; they do not accumulate
separately and cannot refer to other derived values. Member and derived names
must be nonempty strings and must not collide. Collections cannot be nested,
and there is no automatic total or collection-wide Boolean event.

Using the same ``eating`` factory, keep the two color counts and their sum:

.. code-block:: python

   from npc_gym.monitors import MultiMonitor

   vegan = MultiMonitor(
       {"blue": eating("blue"), "orange": eating("orange")},
       derived={"total": lambda counts: counts["blue"] + counts["orange"]},
   )
   vegan.reset(MonitorInput(frozenset()))
   both_colors = MonitorInput(frozenset({"blue", "orange"}))
   assert vegan.update(both_colors) == {
       "blue": True, "orange": True,
   }
   assert vegan.counts == {"blue": 1, "orange": 1, "total": 2}

Here eating both colors increases the total by **two**. This is a sum of counts,
whereas the OR combination above counts once. Choose whichever matches the
question you want to measure.

Rules that remember earlier steps
---------------------------------

Regular expressions describe sequences of inputs. Each bracketed condition
matches one input, and the expression must match the **whole history being
examined**. Conditions not mentioned in a bracket may be either true or false.
Concatenation requires consecutive inputs.

For example, with prefix reporting, ``.* [danger] [fight]`` counts fighting
immediately after danger. The ``.*`` permits any history before that pair.

.. list-table:: Reading a regex
   :header-rows: 1
   :widths: 35 65

   * - Expression
     - Meaning
   * - ``[danger & fight]``
     - Danger and fighting on the same input.
   * - ``[danger] [fight]``
     - Danger on one input, fighting on the next.
   * - ``.``
     - Any one input.
   * - ``.*``
     - Any number of inputs, including none.
   * - ``[danger]*``
     - Zero or more consecutive danger inputs.
   * - ``[danger]?``
     - An optional danger input.
   * - ``[danger] | [fight]``
     - Either danger or fighting.

The `syntax reference <monitor_specifications.rst>`__ covers the full language,
compiler options, and construction from LTLf formulas.

.. _monitor-policies:

What happens after a match?
---------------------------

An automaton monitor first processes the current input, then tests whether the
resulting history matches its specification. ``reporting`` decides whether that
match counts and which history to retain. ``from_regex`` and ``from_ltlf`` use
the same policies:

.. list-table:: Reporting policies
   :header-rows: 1
   :widths: 20 80

   * - Policy
     - Behavior
   * - ``prefix`` (default)
     - Return ``True`` for every matching history, including consecutive matches. Keep the history.
   * - ``restart``
     - Return ``True`` on a match, then clear the history without replaying this input. Keep the count.
   * - ``episode_end``
     - Return ``True`` only if the history matches and ``terminated or truncated`` is ``True``. Keep the history.

For example, ``.* [blue] [blue]`` matches two consecutive blue-eating inputs.
With ``prefix``, three blue inputs produce events on the second and third.
With ``restart``, they produce only the event on the second: that input is
already used, so the third starts a new pair.

Without a counted match the result is ``False`` and history is retained. The
history starts at episode reset, or immediately after the most recent match
for ``restart``. The two different kinds of reset are explained in
`Reset and ownership <#monitor-contracts>`__.

.. _monitor-wrapper:

Using monitors with an environment
----------------------------------

``MonitorWrapper(env, *, monitors={name: factory})`` calls each no-argument
factory to obtain a fresh, independent monitor. The environment must provide
``info["labels"]`` on reset and every step as an iterable of hashable labels;
bare strings and bytes are rejected. Built-in environments already provide
labels; `LabelingWrapper <labeling_functions.rst>`__ adds them to other environments.

On reset the wrapper calls each monitor's ``reset``; on step it calls ``update``
with that step's labels and ending flags. Results distinguish occurrences from counts:

.. code-block:: text

   info["monitors"]["occurrences"][name]          # bool for a Simple/ComplexMonitor
   info["monitors"]["occurrences"][name][member]  # bool for a MultiMonitor member
   info["monitors"]["counts"][name][member]       # cumulative or derived count

A ``SimpleMonitor`` or ``ComplexMonitor`` uses ``member="count"`` only in ``counts``.
A ``MultiMonitor`` uses its member names, with derived names appearing only in ``counts``. Keys stay
fixed across steps and episodes. Both mappings are detached snapshots; the
inner environment must leave ``info["monitors"]`` unused. Observations and
rewards pass through unchanged.

For example, record Merchant's built-in danger count:

.. code-block:: python

   import gymnasium as gym
   import npc_gym
   from npc_gym.monitors import make_builtin_monitor
   from npc_gym.wrappers import MonitorWrapper

   env = MonitorWrapper(
       gym.make("npc_gym/Merchant-v2"),
       monitors={"danger": lambda: make_builtin_monitor("merchant/danger-v0")},
   )
   observation, info = env.reset(seed=0)
   print(info["monitors"]["counts"]["danger"]["count"])
   observation, reward, terminated, truncated, info = env.step(0)
   print(info["monitors"]["occurrences"]["danger"])
   env.close()

The lambda creates a separate history and count for this environment. The first
print reads the count after reset; the second reads the Boolean event from the step.

Use one ``MonitorWrapper`` per environment. Put labeling and time limits inside it,
so it sees their labels and final-step flags. Put vector environments that reset
automatically outside it. After an episode ends—or processing fails—reset before
stepping again. A step that already happened cannot be undone or replayed.

To record these counts with `Evaluation <evaluation.rst>`__ or the SB3 callback, select
``monitor_source="wrapper"``. Alternatively, give evaluation monitor factories
and let it run the monitors itself. Custom monitors need no global registration.

.. _monitor-contracts:

Reset and ownership
-------------------

All monitor types are available from ``npc_gym.monitors``; ``Monitor`` is the
type alias ``ComplexMonitor | MultiMonitor``. Their ``reset(input)`` method has
the same return type as ``update(input)``, but starts a new episode:

1. Clear every component's history and set every event count to zero.
2. Process the supplied input for components with ``consume_initial=True``
   (the default). Those with ``consume_initial=False`` skip it and return ``False``,
   even when the component is a negated expression.
3. Return the initial Boolean event, or the mapping of member events for a
   ``MultiMonitor``. Derived counts are computed from the resulting member counts.

Thus reset can leave a count of one if the initial input matches. It is distinct
from the ``restart`` reporting policy, which clears an automaton's history
after an event while preserving its count. Boolean operands must use the same
initial-input convention; independent collection members may differ.

For example, this monitor counts an initial blue event, then starts over with
an empty initial input:

.. code-block:: python

   initial_blue = eating("blue")
   assert initial_blue.reset(MonitorInput(frozenset({"blue"}))) is True
   assert initial_blue.count == 1
   assert initial_blue.reset(MonitorInput(frozenset())) is False
   assert initial_blue.count == 0

No event is counted before an input is processed, even if a regex accepts the
empty history. A final input is processed normally: ending an episode neither
adds another input nor automatically fails an unfinished obligation. Such a
failure must be part of the rule. Standalone monitors do not enforce episode
boundaries; ``MonitorWrapper`` requires a new reset after termination or truncation.

**Share components within one monitor, not between independent environments.**
The first reset or update assigns the whole group to its outer monitor or
collection. A collection rejects the same member instance under multiple names;
sharing dependencies between distinct expressions is supported. You may inspect
its components afterwards, but call ``update`` and ``reset`` on the outer monitor.
Create fresh monitors for each independent environment or evaluation. Cyclic dependencies are rejected.

Writing a custom detector
~~~~~~~~~~~~~~~~~~~~~~~~~

Most rules can be written as a regex or LTLf formula. For a custom detector,
subclass ``SimpleMonitor`` and implement ``detect(input) -> bool``. If it keeps
history, clear that history in ``reset_history()`` without changing counts or
advancing dependencies. Episode reset calls this hook before optionally consuming
initial labels. The runtime updates the count; the detector only answers whether
the event occurred.
