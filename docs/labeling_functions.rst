Labeling functions
==================

A labeling function describes what happened in an environment step. It might
report “at home,” “ate a blue ghost,” or “collected a frog.” A
`monitor <monitors.rst>`__ can then use those facts to decide whether an event
should be counted.

Labels describe facts, rather than deciding whether an action was allowed.
This lets several monitors interpret the same episode in different ways.
NPC Gym's built-in environments already publish labels in ``info["labels"]``.
Use ``LabelingWrapper`` to add them to another Gymnasium environment.

Labels and transitions
----------------------

A **label** is a value representing a fact that holds for one input. A label set
contains every fact reported for that input; repeating a value adds no information.
Labels must be hashable, so they can be members of a ``frozenset``. Strings,
enum values, and immutable records are suitable choices.

A **transition** describes the action and state change to be labeled. The immutable
``Transition`` record has five fields:

.. list-table:: ``Transition`` record fields
   :header-rows: 1

   * - Field
     - Meaning
   * - ``previous_state``
     - State before the action; ``None`` at reset.
   * - ``action``
     - Action presented to the labeling wrapper; ``None`` at reset.
   * - ``state``
     - State after the action, or the initial state at reset.
   * - ``terminated``
     - Whether the task ended on this step; ``False`` at reset.
   * - ``truncated``
     - Whether an external limit ended the episode; ``False`` at reset.

``LabelingFunction`` specifies the callable contract
``label(transition: Transition) -> frozenset[Label]``. The result describes this
transition, not the union of facts from earlier steps.

Adding labels to an environment
-------------------------------

``LabelingWrapper(env, labeling_function, *, state_extractor=None, mode="extend")``
calls the function once after reset and once after each step:

* On reset, ``previous_state`` and ``action`` are ``None``, ``state`` is the initial
  state, and both ending flags are ``False``.
* On step, the record contains the states before and after the action and the
  ending flags returned by the environment. By default, states are deep copies
  of observations; the action is also copied.

With ``mode="extend"`` (the default), ``info["labels"]`` is the union of your
labels and any existing frozenset of labels. With ``mode="replace"``, it contains
only your function's result. The wrapper preserves observations and rewards.

For example, label whether the player moved in FrozenLake:

.. code-block:: python

   import gymnasium as gym
   from npc_gym.wrappers import LabelingWrapper

   def label_movement(transition):
       moved = (
           transition.previous_state is not None
           and transition.state != transition.previous_state
       )
       return frozenset({"moved"}) if moved else frozenset()

   env = LabelingWrapper(
       gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False),
       label_movement,
   )
   observation, info = env.reset(seed=0)
   assert info["labels"] == frozenset()
   observation, reward, terminated, truncated, info = env.step(2)  # Move right.
   assert "moved" in info["labels"]
   env.close()

The reset produces no movement label because there is no previous state. The
rightward step changes the state and therefore adds ``"moved"``. Strings are
convenient here; built-in environments also use enums and frozen dataclasses,
such as a frog-collection label containing the frog's identifier.

When the monitor should see more than the agent
-----------------------------------------------

Sometimes an agent's observation hides information that a rule needs. For
example, a norm may depend on a location that an observation wrapper conceals.
The optional ``state_extractor(observation, info) -> state`` selects the state
seen by the labeling function. It runs after reset and each step, and must
return an independent snapshot: the wrapper does not copy its result. That
snapshot becomes ``state`` now and ``previous_state`` on the next step.

For example, use Merchant's full state to label whether the agent is at home.
This example includes its own imports:

.. code-block:: python

   import gymnasium as gym
   import npc_gym
   from npc_gym.wrappers import LabelingWrapper

   base_env = gym.make("npc_gym/Merchant-v2")

   def authority_state(observation, info):
       return base_env.unwrapped.labeling_state()

   def label_home(transition):
       return frozenset({"home"}) if transition.state.cell == "H" else frozenset()

   env = LabelingWrapper(base_env, label_home, state_extractor=authority_state)
   observation, info = env.reset(seed=0)
   assert "home" in info["labels"]
   env.close()

After reset, each built-in environment's ``labeling_state()`` returns an
independent, immutable snapshot. Taxi and Merchant provide their state records,
Gardener provides ``GardenerObservation``, and Pacman provides layout, player,
and ghost state. These snapshots are not added to the agent's observation or
``info``. A custom extractor must likewise return a snapshot that later steps
will not change.

Wrapper order
-------------

Place labeling outside any wrapper whose results it needs to describe. In
particular, put ``TimeLimit`` inside ``LabelingWrapper`` if labels depend on
truncation. Put ``MonitorWrapper`` outside labeling so the labels are ready when
the monitor runs; see `monitor integration <monitors.rst#monitor-wrapper>`__. Both belong inside a vector environment
that automatically resets completed episodes.

The action in a ``Transition`` is the action received by ``LabelingWrapper``. If another
wrapper remaps actions, placing it inside labeling records the requested action;
placing it outside labeling records the remapped action. Reward wrappers do not
affect labels, and observation wrappers do not hide data from an explicit state
extractor.
