Environments
============

An environment defines the task's states, actions, rewards, and episode endings.
Importing ``npc_gym`` registers the tasks below for construction with ``gymnasium.make``.
The default time limit counts environment steps and is supplied by Gymnasium's
``TimeLimit`` wrapper. Direct constructors do not impose that time limit.

.. list-table:: Registered tasks
   :header-rows: 1

   * - ID
     - Registered configuration
     - Step limit
     - Rendering
   * - ``npc_gym/StormTaxi-v0``
     - ``StormTaxiEnv``
     - 50
     - ``human``, ``ansi``, ``rgb_array``
   * - ``npc_gym/Merchant-v2``
     - ``MerchantEnv(layout="basic")``
     - 150
     - ``human``, ``rgb_array``
   * - ``npc_gym/Gardener-v0``
     - ``GardenerEnv(size=15)``
     - 1,000
     - ``human``
   * - ``npc_gym/Pacman-v1``
     - ``PacmanEnv(layout="small", features="complete")``
     - 300
     - ``human``, ``rgb_array``
   * - ``npc_gym/PacmanMedium-v1``
     - ``PacmanEnv(layout="medium", features="complete")``
     - 500
     - ``human``, ``rgb_array``
   * - ``npc_gym/PacmanLarge-v1``
     - ``PacmanEnv(layout="large", features="complete")``
     - 800
     - ``human``, ``rgb_array``

Shared behavior
---------------

All environments return ``info["labels"]`` as a frozen set and ``info["episode_metrics"]`` as cumulative task
measurements. Taxi, Merchant, and Gardener also return an ``int8`` ``action_mask``. The mask is advisory: the base
environments retain their documented behavior for valid but masked actions.

Graphical rendering and Pacman image observations require the ``render`` extra, including headless image modes.

Choose ``render_mode`` at construction; create another environment to change modes.
In ``human`` mode, reset and step automatically present frames, including during evaluation.
Taxi paces frames at 4 FPS; Gardener and Pacman use 10 FPS. Merchant uses its configured ``step_delay_ms`` (zero by
default, 500 ms in the paper baseline). Explicit human ``render()`` calls also present and pace a frame.

Headless training, ``rgb_array`` rendering, and Pacman image observations never wait for a human frame rate. Each
renderer owns its drawing surface; human environments share one window. ``close()`` is idempotent and releases only
that environment's resources, leaving other live environments usable. Rendering failures also release the affected
renderer before propagating the error. Reset before reusing an environment after close or a rendering failure.

Storm Taxi
----------

Configuration
~~~~~~~~~~~~~

``StormTaxiEnv(render_mode=None, fickle_passenger=False)`` is available by direct
construction from ``npc_gym.envs``. It composes Gymnasium's dry Taxi dynamics and
renderer with the Storm Taxi rules below. Rain changes weather and normative
obligations, not movement. It has seven actions, 352,000 observations, and no
built-in time limit. Use Gymnasium's ``TimeLimit`` wrapper to limit episodes.
The registered ``npc_gym/StormTaxi-v0`` constructs this class with a 50-step limit.

The environment accepts ``None``, ``ansi``, ``rgb_array`` and ``human``
render modes. Graphical rendering requires the ``render`` extra and adds weather,
flood, hurricane intensity, home and shelter overlays to Gymnasium's 550-by-350
frame. ``rgb_array`` returns a ``uint8`` array with shape ``(350, 550, 3)``;
``human`` presents at four frames per second. Text rendering includes the Warn
action. Rendering does not change the simulation state or random sequence.

Episodes start clear and weather evolves after each action. ``fickle_passenger=True``
gives each episode a 30% chance of one destination change on the first successful
movement with a passenger already aboard.

Actions, observations, and rewards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Actions 0 through 6 are south, north, east, west, pickup, drop-off, and warn.
The ``Discrete(352000)`` observation encodes row, column, passenger, destination,
rain, hurricane stage, home, flood risk, and shelter, with radices
``(5, 5, 5, 4, 2, 11, 4, 2, 4)``. Passenger 4 means inside the taxi.

Clear-weather and active-hurricane actions reward -1, including successful delivery.
While raining without an active hurricane, rewards are -1 normally, -10 for illegal
pickup/drop-off, and +20 for successful delivery. Delivery always terminates.
The episode otherwise ends only through an external time limit. ``episode_metrics``
contains ``success``, which becomes 1 on delivery.

Labels and monitors
~~~~~~~~~~~~~~~~~~~

Labels are:

* actions: ``south``, ``north``, ``east``, ``west``, ``pickup``, ``dropoff``, ``warn``;
* weather: ``rain``, ``hurricane``, ``newHurricane``, ``floodrisk``;
* location: ``atHome``, ``atShelter``, ``atDestination``, and the three corresponding ``toward(...)`` labels;
* passenger: ``hasPassenger``.

Weather, passenger and ``at...`` labels describe the resulting encoded state.
Each ``at...`` label holds exactly when the taxi occupies that location, including
on reset and during stationary actions. A ``toward(...)`` label holds when the
completed movement strictly reduces the shortest route distance to that target.
Both distances use the target in the resulting state. Blocked moves, stationary
actions and resets produce no ``toward(...)`` labels. For example, moving one
cell north into home produces both ``atHome`` and ``toward(atHome)``; warning
there produces ``atHome`` without a direction-of-travel label.

The Emergency monitor counts warning and safety failures in Storm Taxi; see the
`Taxi catalogue <catalogue/taxi.rst>`__.

Source
~~~~~~

Storm Taxi composes Gymnasium's MIT-licensed
`Taxi environment <https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/toy_text/taxi.py>`__,
which is based on Dietterich's `MAXQ Taxi problem <https://doi.org/10.1613/jair.639>`__.
Storm rules and overlays are NPC Gym extensions.

Merchant
--------

Configuration
~~~~~~~~~~~~~

``MerchantEnv(layout="basic", risk_fight=1.0, risk_death=0.0, capacity=5, sunset=28, render_mode=``None``,
step_delay_ms=0)`` accepts the bundled layouts ``basic``, ``cycle``, ``dangerous``, ``possibledanger``,
``possibledanger2``, and ``twist``. ``risk_fight`` is the probability that entering a danger cell triggers an attack;
``risk_death`` is the probability that fighting an attack is fatal. ``capacity`` limits the combined wood and ore inventory,
``sunset`` caps the clock, and ``step_delay_ms`` affects human rendering only.

Actions, observations, and rewards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The seven actions are north, south, east, west, extract, unload, and fight. The tuple observation contains position,
cell type, wood and ore inventory, clock, previous action, and one availability bit per resource. Moving and fighting
successfully earn 0; extracting earns +50; unloading at the market earns +100 per carried resource and terminates;
unloading anywhere else earns -50 per resource and empties inventory; death earns -100 and terminates. An external time
limit may truncate the task. ``episode_metrics`` contains total ``score``, danger and market unload counts, and ``death``.

The action mask excludes walls, extraction without a resource or spare capacity, unloading without cargo or outside
danger/market cells, and fighting outside an attack. During an attack only fight and, with cargo, unload are allowed.
Immediate reversal is excluded whenever another allowed action exists. If reversal is the sole otherwise valid action,
it remains available; for example, an empty-handed arrival at the ``cycle`` market can leave westward.

Extraction at capacity earns zero and advances the clock, leaving inventory and the resource unchanged. This also
applies when a policy ignores the advisory action mask; unloading makes capacity available again.

Labels and monitors
~~~~~~~~~~~~~~~~~~~

Labels are ``atTree``, ``atRock``, ``atHome``, ``atMarket``, ``atDanger``, ``attack``, ``sundown``, ``hasWood``,
``hasOre``, plus one label for each action. Five registered norms cover danger, delivery, environmentally friendly
extraction, an evolving extraction rule, and pacifism; see `Monitor catalogue <monitor_catalogue.rst#monitor-catalogue>`__.

Source
~~~~~~

This environment adapts the Travelling Merchant case study in Neufeld, Bartocci, and Ciabattoni,
`On Normative Reinforcement Learning via Safe Reinforcement Learning
<https://doi.org/10.1007/978-3-031-21203-1_5>`_. The rules above describe the NPC Gym implementation and take
precedence where the published case study differs.

Gardener
--------

Configuration
~~~~~~~~~~~~~

``GardenerEnv(size=15, grass_respawn=50, puddle_respawn=20, score_limit=300, frog_freeze=5, render_mode=None)``
generates a seeded square grid. Entity counts scale from ``size``: 4% grass, 2% puddles, 1% frogs, and 30% walls, each
with at least one grass, puddle, and frog. Parameters control respawn intervals, the terminal score, and the time for
which a drainage event freezes nearby frogs.

Actions, observations, and rewards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Actions 0 through 4 are right, up, left, down, and stay. The flat tuple observation contains the agent and entity
positions, entity state and timers, walls, and current-step collection/drainage events. Active grass earns +10;
draining a full adjacent puddle earns +5; every step then costs 0.1. The task score is capped at ``score_limit`` even
when the final uncapped reward crosses it, and reaching the limit terminates. An external time limit may truncate the
task. ``episode_metrics`` contains the capped ``score``.

Labels and monitors
~~~~~~~~~~~~~~~~~~~

Labels are the five action labels, ``permittedCollect`` when a right action collects a frog, identified
``FrogCollected(frog_id)`` events, and identified ``PuddleDrained(puddle_id, nearby_frog_ids)`` events. The supplied Collect One monitor
counts episodes that end without collecting a frog; see the
`Gardener catalogue <catalogue/gardener.rst>`__.

Source
~~~~~~

Gardener was introduced by Adam and Eiter in `ASP-Driven Emergency Planning for Norm Violations in Reinforcement
Learning <https://doi.org/10.1609/aaai.v39i14.33619>`_. NPC Gym's grass, puddle, scoring, and norm dynamics are its own
variant and are documented above.

Pacman
------

Pacman is a turn-based maze chase with original square mazes and minimalist graphics.
Clear every food pellet while avoiding dangerous ghosts; power pellets temporarily
make ghosts edible. The same full game state supports vectors, pixels, and custom observations.
Ghosts retain their blue, orange, green or pale purple body color when frightened;
a dark zigzag mouth marks that state. Their body brightness also differs in grayscale,
so images preserve ghost identity without requiring memory of an earlier color.

Configuration
~~~~~~~~~~~~~

``PacmanEnv(layout="small", features="complete", dfas=None, render_mode=None,
ghost_behavior="random", ghost_config=None)`` selects the map, observation and ghost behavior.
``GhostConfig(scatter_turns=None, chase_turns=None, random_turn_probability=0.2)``
sets fixed positive turn durations and a probability in [0, 1]. Omitted durations use:

.. list-table:: Bundled maps and schedules
   :header-rows: 1

   * - Map
     - Dimensions
     - Ghosts
     - Scatter / chase turns
   * - ``small``
     - 13 × 13
     - 2
     - 14 / 40
   * - ``medium``
     - 15 × 15
     - 3
     - 18 / 50
   * - ``large``
     - 19 × 19
     - 4
     - 28 / 80

Each map has as many power pellets as ghosts. Walls, food, power pellets and starting
positions are symmetric from left to right. Corridors combine loops, staggered junctions
and dead ends no longer than four moves; some power pellets sit at dead ends.
Ghosts start and respawn in a central, food-free enclosure with one opening onto a junction.
It is ordinary traversable floor: Pacman can enter, and ghosts resume moving immediately
after respawning. Pellet dead-end entrances are separated from the respawn area.
Scheduled cycles
use roughly a 7:20 scatter/chase ratio, with shorter cycles on smaller maps.
Durations count environment turns and do not depend on rendering speed.

Pass a ``PacmanLayout`` instead of a name to load your own geometry::

   from npc_gym.envs import PacmanEnv, PacmanLayout, GhostConfig

   layout = PacmanLayout.from_file("my-maze.lay")
   env = PacmanEnv(layout=layout, ghost_behavior="deterministic",
                   ghost_config=GhostConfig(scatter_turns=10, chase_turns=30))

``PacmanLayout.from_text(text, name="custom")`` accepts the same format. Coordinates
start at the bottom left. The Berkeley-compatible symbols are ``%`` (wall), space
(empty), ``.`` (food), ``o`` (power pellet), ``P`` (player), and ``G`` or ``1``–``4``
(ghost spawns). Ghosts sort by marker, with ``G`` equivalent to ``1``, then by x and y.
Maps require one player, distinct player/ghost spawn cells, food, connected open cells
and an enclosing wall boundary. This also applies to directly constructed layouts.
Trailing whitespace and redundant boundary-wall tails are accepted; playable cells are
never silently discarded. External files are loaded explicitly; no Berkeley maps are bundled.
Names other than ``small``, ``medium`` and ``large`` default to 20 scatter / 70 chase turns. There is no built-in ghost-count cap.

Ghost behavior
~~~~~~~~~~~~~~

``random`` chooses uniformly from its legal actions. It can pause. In a corridor it
can continue or pause; at a junction it excludes reversal. At a dead end it can reverse
or pause. With initial Stop heading, Stop is removed if there are multiple exits.
Every nonempty choice consumes a random draw, including a single forced choice.
Frightened ghosts move half a cell, otherwise one cell, per turn.

``deterministic`` uses four personalities in ghost order: direct pursuit, targeting four
cells ahead of the player, targeting twice the two-cell-ahead position minus ghost 1's
position, and pursuit when more than eight Manhattan cells away with retreat otherwise.
Additional ghosts cycle through these personalities. Scatter targets are upper-right,
upper-left, lower-right, lower-left. Choose the move minimizing squared distance to the
target, with ties north, west, south, east. Frightened ghosts maximize distance to the player.

``partly-deterministic`` uses the same rules, but at junctions with multiple admissible
choices it chooses a uniform action with probability ``random_turn_probability``.
Randomness comes only from the environment's seeded generator. Deterministic ghosts
consume no random draws, including while frightened.

The two scheduled modes choose moves from the same pre-movement state. Following and
merging ghosts keep at least one cell of Manhattan distance between their centers after
movement; adjacent cells are allowed, but a half-cell gap is too small. Moves take priority
in ghost ID order only when all required departures can also proceed. A blocked contender
does not reserve a destination. Followers can enter vacated positions, and closed movement
cycles are allowed. Opposing ghosts on the same corridor can pass with a smaller gap,
but never finish a turn at exactly the same position. This is an NPC Gym spacing rule,
not an arcade-accuracy claim.
Waiting ghosts retain their heading and pending reversal while frightened timers continue
counting down. Frightened expiry never rounds a scheduled ghost's position; it finishes any
remaining half-cell before returning to full-cell strides. Random ghosts do not use these
spacing rules.

Scheduled modes start in scatter and repeat fixed intervals without increasing difficulty.
A phase switch takes effect after its last turn. Reverse direction on a phase switch or
power pellet; when between cells, defer reversal until the next center. Otherwise reverse
only at dead ends. The schedule pauses on a power-pellet turn and on every turn that starts
with any frightened ghost, including the final frightened turn. Resume the remaining phase
interval afterward. Frightened timers are per ghost; an eaten ghost respawns immediately.
Random ghosts always return to their own starting position. Scheduled ghosts prefer their
own starting position, but if it is less than one cell from Pacman or another ghost, use
the nearest floor cell with that clearance (breadth-first ties north, south, east, west).
If a densely populated custom map has no such cell, use the nearest unoccupied cell.
Respawn clears the frightened timer.

Actions, observations, and rewards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Actions 0–4 are stop, north, south, east, west. A blocked direction becomes Stop while
retaining heading. Each turn moves the player and resolves collisions. Random ghosts then
move in identity order with collisions after each move; termination stops further moves.
Scheduled ghosts move together after resolving destination conflicts, then resolve collisions
with Pacman in identity order, stopping collision processing at termination.
Power pellets set frightened timers to 40. Ghost timers decrement after movement and before
collision. For random ghosts only, expiry rounds a half-cell position to the nearest
cell, with .5 rounding upward.

Task score changes are -1 per turn, +10 per food pellet, +200 per edible ghost,
+500 for clearing the food, and -500 per dangerous collision. Clearing the final food wins
before dangerous collisions can cause a loss. Winning or losing terminates; only an external
``TimeLimit`` truncates. Use ``npc_gym/Pacman-v1`` for small (300 turns),
``npc_gym/PacmanMedium-v1`` for medium (500), and ``npc_gym/PacmanLarge-v1`` for large (800).
Gymnasium's limit belongs to the registered ID: overriding only ``layout`` does not change it.
Pass ``max_episode_steps=...`` to ``gymnasium.make`` to override the limit, or ``-1`` to disable it.
Directly constructed environments have no limit. To apply the map's recommended limit explicitly::

   from gymnasium.wrappers import TimeLimit

   env = PacmanEnv(layout="large")
   env = TimeLimit(env, max_episode_steps=env.layout.default_episode_steps)

``PacmanLayout.default_episode_steps`` selects 300/500/800 by the names small/medium/large,
and 300 for other names. It is a recommendation for external wrappers, not a simulation rule.

Vector modes are ``complete``, ``complete-distinguish``, ``deep-rl``, ``dfa``,
``dfa-distinguish``, ``essential``, ``essential-na``, ``hungry``, ``labelled``, and
``next-action``. They return float64 vectors. ``dfas`` supplies known automaton names and
finite rewards for complete/DFA modes. Embedded DFA features evaluate their definition
from its initial state on each input; use `restraining bolts <restraining_bolts.rst>`__
for retained automaton history and reward adjustments.

Image modes are ``image-crop`` (84 × 84 RGB), ``image-full`` and
``image-full+<vector-mode>``. Full frames are uint8 arrays of height ``30*(map_height+1)``
and width ``30*(map_width+1)``. Combined modes append a fourth channel containing DFA bits
in its first row, with unused entries 255. They do not encode the entire feature vector.
Crops use a fixed window anchored to the player's position, resize it to 84 × 84, and
pad outside the frame with black. The player appears above the image center; this window
does not center the player sprite.

``ghost_config`` applies only to scheduled behaviors. Random mode accepts and retains
resolved configuration values in ``state().config`` but ignores them during gameplay;
its phase is ``None`` and its countdown is zero.

Random mode returns a ``Box`` observation. Scheduled modes append three float64 values to
vectors: chase flag (0 scatter / 1 chase), remaining phase turns, and paused flag. Images
instead return a ``Dict`` with ``observation`` pixels and ``ghost_mode`` containing those
three values. Frightened status remains per ghost. ``PacmanPixelObservation`` can grayscale
and stack image frames while preserving mode and bolt features in a flat dictionary suitable
for SB3's multi-input policies; its defaults are two 80-by-210 frames.

Custom observations and labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``env.state()`` returns an immutable ``PacmanSnapshot`` after reset. Its ``game`` field is
the authority state: geometry, food, capsules, player, identified ghosts, score, terminal
flags and current events. Additional fields expose resolved controller configuration,
schedule, spawn positions, pending reversals and turn count. It is an observation snapshot,
not an RNG checkpoint. ``labeling_state()`` returns just the authority state. Retained
snapshots cannot change when the environment advances. A custom Gymnasium observation wrapper
can build any desired observation from ``env.unwrapped.state()`` and declare its own space.

Blue and orange mean ghosts 1 and 2. Existing monitors and policy fixes retain that scope;
ghosts 3 and 4 are not implicitly included in Vegan. Labels include actions, eating,
frightened adjacency, score thresholds, west-side and corner propositions. The named
``inSouthEast`` proposition currently identifies the upper-right interior corner.
``episode_metrics`` reports score, blue/orange eating counts, remaining food and win/loss flags.
See `Monitor catalogue <monitor_catalogue.rst#monitor-catalogue>`__.

Manual play uses ``python examples/play_pacman.py --layout small --ghost-behavior deterministic``
with the ``render`` extra installed. Arrows/WASD move, Space stops, P pauses, R restarts,
and Escape exits. It starts paused; selecting a direction begins play.
A blocked turn leaves the current direction unchanged. A pending direction keeps trying
while its key is held, then remains buffered for one second of real time after release.
It takes the first available turn in that time. Pressing another direction replaces the
pending turn. After a buffered turn succeeds or expires,
the most recently pressed held key takes priority. Space, pause, restart and loss of
window focus clear the buffer; losing focus also pauses play.

Source
~~~~~~

The normative Pacman case study originates in Neufeld, Bartocci, Ciabattoni, and Governatori,
`A Normative Supervisor for Reinforcement Learning Agents <https://doi.org/10.1007/978-3-030-79876-5_32>`_,
which builds on the UC Berkeley Pacman projects. The high-level vector features and the norms beyond the
earlier vegan and vegetarian ones follow Neufeld, Engesser, and Tappler,
`Scalable Learning of Challenging Normative Behaviours with Deep RL <https://doi.org/10.24963/KR.2026/99>`_.
The simulation, bundled maps and renderer are NPC Gym implementations; see
`THIRD_PARTY_NOTICES.md <../THIRD_PARTY_NOTICES.md>`__ for the retained paper layout.
