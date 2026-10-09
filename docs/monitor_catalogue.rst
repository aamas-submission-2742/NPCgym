.. _monitor-catalogue:

Monitor catalogue
=================

Choose an environment to find its built-in monitors. Each entry explains what
it counts and gives its ID, type, and recipe: a regex and reporting policy for
a simple monitor, a Boolean function for a complex monitor, or members and
additional totals for a collection.

* `Taxi <catalogue/taxi.rst>`__: warnings and hurricane safety.
* `Merchant <catalogue/merchant.rst>`__: danger, deliveries, fighting, and extraction.
* `Gardener <catalogue/gardener.rst>`__: collecting a frog before the episode ends.
* `Pacman <catalogue/pacman.rst>`__: eating, movement, deadlines, and combinations.

.. toctree::
   :hidden:
   :maxdepth: 1

   catalogue/taxi
   catalogue/merchant
   catalogue/gardener
   catalogue/pacman

Use an ID with ``make_builtin_monitor(id)`` to create a fresh instance.
The `monitor guide <monitors.rst>`__ explains the types, examples, and contracts;
this catalogue is the reference for the supplied rules.

Reading an entry
----------------

The **monitor name** is its display name, such as Hungry Vegan. The **ID** is the
string passed to ``make_builtin_monitor``, such as ``"pacman/hungry-vegan-v0"``.
A **count key** is a name in the result, such as ``"Hungr"``; it need not match
the display name. A simple or composed event monitor exposes one ``count``;
a collection exposes its named member counts and any explicitly listed totals.

Each entry specifies the event that increases a count, whether reset is processed,
and how timing and simultaneous events are handled. A **deadline** is inclusive:
fulfilling the requirement on that input is in time unless the entry explicitly
states otherwise. No unprocessed future step contributes a count after an episode ends.

Regex notation
--------------

The regex notation is explained in `Monitors <monitors.rst>`__ and listed in full in
`Monitor specifications <monitor_specifications.rst>`__. In the tables, the notation *R^n* abbreviates *n* copies
of an expression *R*; zero copies means ``eps``. Named regex abbreviations are defined beside
their recipes. Expand powers and abbreviations before passing them to
``from_regex``.

The shipped definitions are in ``npc_gym.monitors.builtins``; the meanings of
their conditions are in ``npc_gym.monitors.bindings``. The built-ins' simple
detectors use regexes; custom simple monitors can also be constructed from LTLf.
