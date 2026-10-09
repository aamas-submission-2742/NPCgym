import ast
import hashlib
import inspect
import itertools
import json
import textwrap

import pytest

from npc_gym.automata import pacman_dfas
from npc_gym.automata.automaton import AutomatonDefinition, AutomatonRunner, Lifecycle
from npc_gym.envs.pacman.labels import PacmanLabel

EXPECTED_DEFINITIONS = {
    "AllOrNothingDFA": (0, (0, 1, 2), (2,), Lifecycle.MAINTENANCE, "763db0aa57f741e7"),
    "CTDBlueDFA": (0, (0, 1, 2), (2,), Lifecycle.PUNCTUAL, "e25813ea1a4513b2"),
    "CTDBlueDFA3": (0, (0, 1, 2, 3, 4, 5, 6), (2,), Lifecycle.PUNCTUAL, "7d7c0b3ca119a622"),
    "CTDBlueDFA5": (0, (0, 1, 2, 3, 4, 5, 6), (2,), Lifecycle.PUNCTUAL, "97aaa66c3ad6a83c"),
    "CTDOrangeDFA": (0, (0, 1, 2), (2,), Lifecycle.PUNCTUAL, "e25813ea1a4513b2"),
    "CTDOrangeDFA3": (0, (0, 1, 2, 3, 4, 5, 6), (2,), Lifecycle.PUNCTUAL, "7d7c0b3ca119a622"),
    "CTDOrangeDFA5": (0, (0, 1, 2, 3, 4, 5, 6), (2,), Lifecycle.PUNCTUAL, "97aaa66c3ad6a83c"),
    "EarlyBirdDFA1": (0, (0, 1, 2), (2,), Lifecycle.ACHIEVEMENT, "1fddc2387564b9b6"),
    "EarlyBirdDFA2": (0, (0, 1, 2), (2,), Lifecycle.ACHIEVEMENT, "a4fd6c7a48a315df"),
    "EarlyBirdFulfillmentDFA": (0, (0, 1, 2, 3, 4), (3,), Lifecycle.ACHIEVEMENT, "1f6aba7505fbead9"),
    "EarlyBirdFulfillmentDFA2": (0, (0, 1, 2, 3, 4), (3,), Lifecycle.ACHIEVEMENT, "ab971771f310828b"),
    "ErrandDFA1a": (0, (0, 1, 2), (2,), Lifecycle.ACHIEVEMENT, "a4fd6c7a48a315df"),
    "ErrandDFA1b": (0, (0, 1, 2, 3), (2,), Lifecycle.ACHIEVEMENT, "eba1bffc0889ff74"),
    "ErrandDFA2a": (0, (0, 1, 2), (2,), Lifecycle.ACHIEVEMENT, "eede69d9565dfb70"),
    "ErrandDFA2b": (0, (0, 1, 2, 3), (2,), Lifecycle.ACHIEVEMENT, "314efd8c1378fe17"),
    "OblBlueDFA": (0, (0, 1, 2, 3), (3,), Lifecycle.PUNCTUAL, "25aceacb1d8dc8a5"),
    "OneTasteDFA": (0, (0, 1, 2), (2,), Lifecycle.ACHIEVEMENT, "501eebb7309d9dcc"),
    "PermBlueDFA": (0, (0, 1), (1,), Lifecycle.PUNCTUAL, "1695b35eda9e3149"),
    "PowerPelletDFA": (0, (0, 1), (1,), Lifecycle.PUNCTUAL, "1695b35eda9e3149"),
    "ScoreHelper": (0, (0, 1, 2, 3, 4), (), Lifecycle.HELPER, "3cca7054d865ac51"),
    "TrappedDFA": (0, (0, 1, 2), (1,), Lifecycle.MAINTENANCE, "8c3119fb0e292e95"),
    "VegBlueDFA": (0, (0, 1), (1,), Lifecycle.PUNCTUAL, "1695b35eda9e3149"),
    "VegOrangeDFA": (0, (0, 1), (1,), Lifecycle.PUNCTUAL, "1695b35eda9e3149"),
    "VeganPowerDFA": (0, (0, 1), (1,), Lifecycle.PUNCTUAL, "1695b35eda9e3149"),
}


def definitions():
    return {
        name: value
        for name, value in inspect.getmembers(pacman_dfas, inspect.isclass)
        if issubclass(value, AutomatonDefinition) and value is not AutomatonDefinition
    }


def transition_digest(definition):
    tree = ast.parse(textwrap.dedent(inspect.getsource(type(definition).transition)))
    labels = sorted(
        {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    )
    reached = []
    for state in definition.states:
        for included in itertools.product((False, True), repeat=len(labels)):
            label_set = frozenset(label for label, present in zip(labels, included, strict=True) if present)
            reached.append(definition.transition(label_set, state))
    return hashlib.sha256(json.dumps(reached, separators=(",", ":")).encode()).hexdigest()[:16]


def test_every_hand_authored_definition_is_characterized():
    assert definitions().keys() == EXPECTED_DEFINITIONS.keys()


@pytest.mark.parametrize("name", sorted(EXPECTED_DEFINITIONS))
def test_definition_matches_its_recorded_graph(name):
    definition = definitions()[name](frozenset(PacmanLabel))
    initial, states, finals, lifecycle, digest = EXPECTED_DEFINITIONS[name]
    assert (definition.initial_state, definition.states) == (initial, states)
    assert tuple(sorted(definition.final_states)) == finals
    assert definition.lifecycle is lifecycle
    assert transition_digest(definition) == digest


def test_a_specialized_definition_seals_only_once_its_own_initializer_returns():
    class CountingVegBlue(pacman_dfas.VegBlueDFA):
        def __init__(self, alphabet, reward=0):
            super().__init__(alphabet, reward)
            self.note = "specialized"

    definition = CountingVegBlue(frozenset(PacmanLabel))

    assert definition.note == "specialized"
    assert AutomatonRunner(definition).step({PacmanLabel.EAT_BLUE_GHOST}) == 1
    with pytest.raises(AttributeError, match="immutable"):
        definition.note = "changed"


def test_definitions_are_immutable_and_runners_have_independent_state():
    definition = pacman_dfas.OblBlueDFA(frozenset(PacmanLabel))
    first = AutomatonRunner(definition)
    second = AutomatonRunner(definition)
    assert first.step({PacmanLabel.ADJACENT_BLUE_GHOST}) == 1
    assert first.state == 1
    assert second.state == 0
    with pytest.raises(AttributeError, match="immutable"):
        definition.reward = 10


@pytest.mark.parametrize(
    ("definition", "start", "labels", "reached", "retained"),
    [
        (pacman_dfas.CTDBlueDFA, 1, frozenset(), 2, 0),
        (pacman_dfas.EarlyBirdDFA1, 1, {PacmanLabel.SCORE_GREATER_100}, 2, 0),
        (pacman_dfas.TrappedDFA, 2, {PacmanLabel.SCORE_0}, 1, 2),
    ],
)
def test_lifecycle_strategy_preserves_legacy_reset_behavior(definition, start, labels, reached, retained):
    runner = AutomatonRunner(definition(frozenset(PacmanLabel)))
    runner.state = start
    assert runner.step(labels) == reached
    assert runner.state == retained


def test_runner_accepts_arbitrary_hashable_labels():
    marker = ("event", 3)
    definition = pacman_dfas.VegBlueDFA({marker, PacmanLabel.EAT_BLUE_GHOST})
    assert AutomatonRunner(definition).step({marker}) == definition.initial_state


def test_fulfillment_runner_resets_on_its_auxiliary_violation():
    runner = AutomatonRunner(pacman_dfas.EarlyBirdFulfillmentDFA(frozenset(PacmanLabel)))
    runner.step({PacmanLabel.SCORE_0})
    assert runner.state == 1
    runner.step({PacmanLabel.SCORE_GREATER_100})
    assert runner.state == 0
    assert runner.violation_runner is not None
    assert runner.violation_runner.state == 0


def test_abstract_definition_requires_a_transition_implementation():
    with pytest.raises(TypeError, match="abstract method.*transition"):
        AutomatonDefinition(frozenset(PacmanLabel))


@pytest.mark.parametrize("definition_class", [pacman_dfas.CTDBlueDFA3, pacman_dfas.CTDOrangeDFA3])
@pytest.mark.parametrize("state", [5, 6])
def test_legacy_unused_states_have_no_transition_and_are_rejected_by_runners(definition_class, state):
    definition = definition_class(frozenset(PacmanLabel))
    assert state in definition.states
    assert definition.transition(frozenset(), state) is None
    runner = AutomatonRunner(definition)
    runner.state = state
    with pytest.raises(ValueError, match="transition returned undeclared state None"):
        runner.step(frozenset())
    assert runner.state == state


@pytest.mark.parametrize(
    ("definition_class", "trigger"),
    [
        (pacman_dfas.CTDBlueDFA3, PacmanLabel.EAT_BLUE_GHOST),
        (pacman_dfas.CTDOrangeDFA3, PacmanLabel.EAT_ORANGE_GHOST),
    ],
)
def test_three_step_tables_never_reach_their_unused_states(definition_class, trigger):
    definition = definition_class(frozenset(PacmanLabel))
    inputs = [
        frozenset(),
        frozenset({trigger}),
        frozenset({PacmanLabel.STAYED_STILL}),
        frozenset({trigger, PacmanLabel.STAYED_STILL}),
    ]
    reached = {definition.initial_state}
    pending = [definition.initial_state]
    while pending:
        state = pending.pop()
        for labels in inputs:
            successor = definition.transition(labels, state)
            assert successor in definition.states
            if successor not in reached:
                reached.add(successor)
                pending.append(successor)
    assert reached == {0, 1, 2, 3, 4}
