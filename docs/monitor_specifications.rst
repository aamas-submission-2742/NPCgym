Monitor specifications
======================

This is the reference for writing regex and LTLf specifications. For a first
example, the three monitor types, or reset behavior, start with `Monitors <monitors.rst>`__.
For existing rules, use the separate `Monitor catalogue <monitor_catalogue.rst>`__.

Constructing a simple monitor
-----------------------------

``from_regex(expression, propositions=..., reporting="prefix", consume_initial=True, minimize=True)``
returns an ``AutomatonMonitor``, which is a ``SimpleMonitor``. Each key in
``propositions`` names a condition in the expression; its function receives a
``MonitorInput`` and returns a Boolean. Supply every condition used by the
expression. Extra bindings are allowed. The functions must depend only on their
input, leaving any memory to the monitor.

A **proposition** is a named condition evaluated on one ``MonitorInput``. A binding
is the function that supplies its Boolean value. A regex describes a sequence
of those Boolean valuations: the true-or-false value of every proposition on each input.
``[a & b]`` requires both conditions on one input; ``[a] [b]`` requires two
inputs in order. Unmentioned conditions can be true or false. The expression
matches the whole history: ``.* [a] [b]`` requires its final two inputs to
satisfy a and b, while ``.* [a] [b] .*`` continues to match later inputs too.
See `What happens after a match? <monitors.rst#monitor-policies>`__ for what the monitor does after a match.

Regex syntax
------------

.. code-block:: text

   regex         := concatenation ("|" concatenation)*
   concatenation := repetition+
   repetition    := position ("*" | "+" | "?")?
   position      := "[" boolean "]" | "." | "eps" | "(" regex ")"
   boolean       := conjunction ("|" conjunction)*
   conjunction   := unary ("&" unary)*
   unary         := "!"* (identifier | "true" | "false" | "(" boolean ")")

Identifiers are case-sensitive ASCII ``[A-Za-z_][A-Za-z0-9_]*``; ``true`` and ``false`` are reserved Boolean
constants. ``eps`` denotes the empty word outside tests and is an ordinary identifier inside tests. Whitespace
between tokens is insignificant; it does not join split identifiers. For example, ``eps eps`` concatenates two
empty words, while ``epseps`` is not a regex position.

``.`` consumes any valuation, ``*`` means zero or more repetitions, ``+`` one or more, and ``?`` zero or one.
Postfix repetition binds before concatenation, which binds before regex alternation. Within tests, ``!`` binds
before ``&``, then ``|``. Parentheses override precedence. Repeated postfix operators require grouping, such as
``([a]*)?``; lazy quantifiers, character classes, escapes, anchors, and bounded repetitions are not supported.
Use ``eps`` for an empty word and ``[false]`` for an empty language; empty source text and empty alternatives are errors.

Compilation limits
------------------

Compilation turns a rule into a finite-state machine that summarizes its history.
The limits below bound the expression size and the work needed to build that machine.
An NFA is the intermediate machine, which can have several possible next states;
a DFA is the final deterministic machine, which selects exactly one next state for each input.

``from_regex`` and ``compile_regex`` accept an immutable ``RegexLimits`` value; omitted limits use these defaults:

.. list-table:: Compilation budgets
   :header-rows: 1

   * - Field
     - Default
     - Counts
   * - ``max_length``
     - 10,000
     - Source characters, including whitespace.
   * - ``max_nesting``
     - 64
     - Open regex groups, Boolean tests, and Boolean groups combined.
   * - ``max_atoms``
     - 10
     - Distinct proposition identifiers.
   * - ``max_nfa_states``
     - 512
     - Intermediate NFA states.
   * - ``max_dfa_states``
     - 256
     - DFA states before minimization, including the initial state and explicit sink.
   * - ``max_transitions``
     - 8,192
     - Guarded transitions across all DFA states.

For example, ``compile_regex(expression, limits=RegexLimits(max_dfa_states=512))`` increases the DFA-state budget.
All budgets must be positive non-Boolean integers. ``RegexSyntaxError`` is a ``ValueError`` with ``position``
(zero-based offset), ``line``, and ``column`` (one-based). ``RegexCompilationError`` reports the exceeded budget;
very deeply nested expressions can also reach the Python parser recursion capacity after increasing the nesting
budget. Compilation failures never return a partial DFA. Invalid argument types raise ``TypeError``.

LTLf formulas
-------------

LTLf describes properties of finite sequences using temporal operators. For
example, ``F danger`` says that danger occurs somewhere in the sequence. To use
it, install the ``ltlf`` extra (``python -m pip install '.[ltlf]'`` from a checkout) and the
`MONA executable <https://www.brics.dk/mona/download.html>`_ on PATH.
Regex monitors and the built-in catalogue need neither dependency.

.. code-block:: python

   from npc_gym.monitors import MonitorInput
   from npc_gym.monitors.ltlf import from_ltlf

   monitor = from_ltlf(
       "F danger",
       propositions={"danger": lambda step: "danger" in step.labels},
       reporting="episode_end",
   )
   monitor.reset(MonitorInput(frozenset()))
   monitor.update(MonitorInput(frozenset({"danger"})))
   assert monitor.count == 0
   assert monitor.update(MonitorInput(frozenset(), terminated=True))
   assert monitor.count == 1

This counts one event at the end of an episode that included danger. With
``prefix`` reporting, the same formula would count on every input from the first
danger onward, because “danger has occurred” remains true. To test danger on
just the current input, use ``F(danger & !(X true))``: ``!(X true)`` identifies
the last position in the sequence examined so far.

Supported temporal operators include strong next ``X``, weak next ``WX``,
eventually ``F``, always ``G``, and until ``U``, together with Boolean operators.
Proposition names must match ``[a-z][a-z0-9_]*``. ``from_ltlf`` returns the same
kind of ``SimpleMonitor`` as ``from_regex``, with the same reporting policies and
initial-input option. Either kind can take part in Boolean combinations and
collections.

Working with compiled automata
------------------------------

This section is for callers who need to reuse a compiled definition or supply a
different compiler. Ordinary monitor construction needs only the factories above.

Use ``compile_regex(expression)`` when you want a reusable ``CompiledDFA``
definition, then construct an ``AutomatonMonitor`` with that definition and its
proposition bindings. The definition can be shared; each monitor keeps its own
current state and count. ``monitor.state`` gives the automaton state retained
for the next input, including a restart after a match when that policy is used.

Regex compilation sorts proposition names and numbers states deterministically.
Its transition conditions use ``0`` for false, ``1`` for true, and ``X`` for either;
the conditions are disjoint and cover every input.

Regex and LTLf monitor factories minimize their DFAs by default: unreachable
states are removed, and states that accept exactly the same future input
sequences are merged. This preserves every event, count, and reporting policy.
Minimization considers all Boolean valuations, without assuming additional
relationships between domain propositions. The initial state is numbered 0;
the remaining state IDs are deterministic consecutive integers.

Set ``minimize=False`` in ``compile_regex``, ``from_regex``, ``from_ltlf`` or
``make_builtin_monitor`` to retain the compiler's original states. Regex
compilation then retains its explicit rejecting sink even if unreachable.
Compilation budgets apply before minimization; a small final DFA does not
bypass limits on the intermediate construction. For an existing definition,
``definition.minimized()`` returns a minimal ``CompiledDFA`` without modifying
the original; it may return the same object if already in the resulting form.
Direct ``CompiledDFA`` and ``AutomatonMonitor`` construction preserves the
supplied states. Runtime steps never perform minimization.

Compilation can consider up to 2 raised to ``max_atoms`` combinations of truth
values, so raising that limit can substantially increase time and memory use.

For a different LTLf compiler, pass ``compiler=...`` to ``from_ltlf``. It must
provide ``compile(formula)`` returning a ``CompiledDFA``. The default
``MonaCompiler(executable="mona", timeout=30, minimize=True)`` runs in a temporary directory and
cleans up after success or failure. Optional dependencies are loaded only when
compilation is requested. With an injected compiler, ``minimize=False`` preserves
exactly the definition that compiler returns; it cannot undo its own minimization.

A missing Python dependency raises ``ImportError``; a missing MONA executable raises
``FileNotFoundError``. Invalid formulas or bindings raise ``ValueError`` or ``TypeError``.
Compiler failures, timeouts, and invalid automata raise ``LTLfCompilationError``.
There is no automatic fallback to another compiler.
