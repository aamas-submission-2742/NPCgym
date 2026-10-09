"""Hand-authored Pacman automata definitions.

These definitions encode the historical transition tables; they are not claimed
to have been generated from LTLf formulas. Alphabets contain hashable labels;
transitions inspect label membership and default to the initial state when no
state is supplied. They return None where a historical table has no branch,
including the unreachable states 5 and 6 of CTDBlueDFA3 and CTDOrangeDFA3.
AutomatonRunner rejects such results because None is not a declared state.
"""

from collections.abc import Container, Iterable

from npc_gym.automata.automaton import AutomatonDefinition as DFA
from npc_gym.automata.automaton import Label, State

# NOTE: these correspond to individual norms, not entire monitors


# Norm: F(eatPowerPellet)
# Violation specification: F(eatPowerPellet)
class VeganPowerDFA(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, -reward)
        self.states = [0, 1]
        self.final = [1]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "eatPowerPellet" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            return 1
        return None


# Norm: F(eatBlueGhost)
# Violation specification: F(eatBlueGhost)
class VegBlueDFA(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, -reward)
        self.states = [0, 1]
        self.final = [1]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "eatBlueGhost" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            return 1
        return None


# Norm: F(eatOrangeGhost)
# Violation specification: F(eatOrangeGhost)
class VegOrangeDFA(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, -reward)
        self.states = [0, 1]
        self.final = [1]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "eatOrangeGhost" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            return 1
        return None


# Norm: P(eatBlueGhost)
# Violation specification: F(eatBlueGhost)
class PermBlueDFA(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, reward)
        self.states = [0, 1]
        self.final = [1]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "eatBlueGhost" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            return 1
        return None


# Norm: O(eatBlueGhost [within 3 steps] | adjacentBlueGhost)
# Violation specification:
class OblBlueDFA(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, -reward)
        self.states = [0, 1, 2, 3]
        self.final = [3]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "adjacentBlueGhost" in inpt and "eatBlueGhost" not in inpt:
                return 1
            else:
                return 0
        if state == 1:
            if "eatBlueGhost" in inpt:
                return 0
            else:
                return 2
        if state == 2:
            if "eatBlueGhost" in inpt:
                return 0
            else:
                return 3
        if state == 3:
            return 3
        return None


# Norm: F(eatPowerPellet)
# Violation specification: F(eatPowerPellet)
class PowerPelletDFA(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, -reward)
        self.states = [0, 1]
        self.final = [1]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "eatPowerPellet" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            return 1
        return None


# TODO: generate DFAs for more complex norms [eventually incorporate API]


# TODO: add AllOrNothing permission to documentation; add temporal permission to set of criteria
# TODO: define permission specifications for temporal permissions!!!!
# Norm: P^M_\bot(eatBlueGhost v eatOrangeGhost | eatBlueGhost v eatOrangeGhost )
# Permission specification: F((eatBlueGhost | eatOrangeGhost) & (T U X(eatBlueGhost | eatOrangeGhost)))
class AllOrNothingDFA(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, reward, reset="maintenance", permission=True)
        self.states = [0, 1, 2]
        self.final = [2]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "eatBlueGhost" in inpt or "eatOrangeGhost" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            if "eatBlueGhost" in inpt or "eatOrangeGhost" in inpt:
                return 2
            else:
                return 1
        if state == 2:
            return 2
        return None


class OneTasteDFA(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, reward, reset="achievement", permission=True)
        self.states = [0, 1, 2]
        self.final = [2]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "score0" in inpt and ("eatBlueGhost" in inpt or "eatOrangeGhost" in inpt):
                return 2
            elif "score0" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            if "eatBlueGhost" in inpt or "eatOrangeGhost" in inpt:
                return 2
            else:
                return 1
        if state == 2:
            return 2
        return None


# Norm: O(X(Stop) | eatBlueGhost)
# G(eatBlueGhost -> (X(stop) | X2(stop) | X3(stop) | X4(stop) | X5(stop)) => F(eatBlueGhost) & X(!stop) & X2(!stop)...) [ands inside otherwise weak next]
# Violation specification: F(X(!Stop) & eatBlueGhost)
class CTDBlueDFA(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, -reward, reset="punctual")
        self.states = [0, 1, 2]
        self.final = [2]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "eatBlueGhost" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            if "stayedStill" not in inpt:
                return 2
            elif "eatBlueGhost" in inpt:
                return 1
            else:
                return 0
        if state == 2:
            return 2
        return None


# Norm: O(X(Stop) | eatOrangeGhost)
# Violation specification: F(X(!Stop) & eatBlueGhost)
class CTDOrangeDFA(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, -reward, reset="punctual")
        self.states = [0, 1, 2]
        self.final = [2]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "eatOrangeGhost" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            if "stayedStill" not in inpt:
                return 2
            elif "eatOrangeGhost" in inpt:
                return 1
            else:
                return 0
        if state == 2:
            return 2
        return None


# Norm: O(X(Stop) | eatBlueGhost)
# G(eatBlueGhost -> (X(stop) | X2(stop) | X3(stop) | X4(stop) | X5(stop)) => F(eatBlueGhost) & X(!stop) & X2(!stop)...) [ands inside otherwise weak next]
# Violation specification: F(X(!Stop) & eatBlueGhost)
class CTDBlueDFA3(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, -reward, reset="punctual")
        self.states = [0, 1, 2, 3, 4, 5, 6]
        self.final = [2]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "eatBlueGhost" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            if "stayedStill" not in inpt:
                return 2
            elif "eatBlueGhost" in inpt:
                return 1
            else:
                return 3
        if state == 3:
            if "stayedStill" not in inpt:
                return 2
            elif "eatBlueGhost" in inpt:
                return 1
            else:
                return 4
        if state == 4:
            if "stayedStill" not in inpt:
                return 2
            elif "eatBlueGhost" in inpt:
                return 1
            else:
                return 0
        if state == 2:
            return 2
        return None


# Norm: O(X(Stop) | eatOrangeGhost)
# Violation specification: F(X(!Stop) & eatBlueGhost)
class CTDOrangeDFA3(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, -reward, reset="punctual")
        self.states = [0, 1, 2, 3, 4, 5, 6]
        self.final = [2]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "eatOrangeGhost" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            if "stayedStill" not in inpt:
                return 2
            elif "eatOrangeGhost" in inpt:
                return 1
            else:
                return 3
        if state == 3:
            if "stayedStill" not in inpt:
                return 2
            elif "eatOrangeGhost" in inpt:
                return 1
            else:
                return 4
        if state == 4:
            if "stayedStill" not in inpt:
                return 2
            elif "eatOrangeGhost" in inpt:
                return 1
            else:
                return 0
        if state == 2:
            return 2
        return None


# Norm: O(X(Stop) | eatBlueGhost)
# G(eatBlueGhost -> (X(stop) | X2(stop) | X3(stop) | X4(stop) | X5(stop)) => F(eatBlueGhost) & X(!stop) & X2(!stop)...) [ands inside otherwise weak next]
# Violation specification: F(X(!Stop) & eatBlueGhost)
class CTDBlueDFA5(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, -reward, reset="punctual")
        self.states = [0, 1, 2, 3, 4, 5, 6]
        self.final = [2]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "eatBlueGhost" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            if "stayedStill" not in inpt:
                return 2
            elif "eatBlueGhost" in inpt:
                return 1
            else:
                return 3
        if state == 3:
            if "stayedStill" not in inpt:
                return 2
            elif "eatBlueGhost" in inpt:
                return 1
            else:
                return 4
        if state == 4:
            if "stayedStill" not in inpt:
                return 2
            elif "eatBlueGhost" in inpt:
                return 1
            else:
                return 5
        if state == 5:
            if "stayedStill" not in inpt:
                return 2
            elif "eatBlueGhost" in inpt:
                return 1
            else:
                return 6
        if state == 6:
            if "stayedStill" not in inpt:
                return 2
            elif "eatBlueGhost" in inpt:
                return 1
            else:
                return 0
        if state == 2:
            return 2
        return None


# Norm: O(X(Stop) | eatOrangeGhost)
# Violation specification: F(X(!Stop) & eatBlueGhost)
class CTDOrangeDFA5(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, -reward, reset="punctual")
        self.states = [0, 1, 2, 3, 4, 5, 6]
        self.final = [2]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "eatOrangeGhost" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            if "stayedStill" not in inpt:
                return 2
            elif "eatOrangeGhost" in inpt:
                return 1
            else:
                return 3
        if state == 3:
            if "stayedStill" not in inpt:
                return 2
            elif "eatOrangeGhost" in inpt:
                return 1
            else:
                return 4
        if state == 4:
            if "stayedStill" not in inpt:
                return 2
            elif "eatOrangeGhost" in inpt:
                return 1
            else:
                return 5
        if state == 5:
            if "stayedStill" not in inpt:
                return 2
            elif "eatOrangeGhost" in inpt:
                return 1
            else:
                return 6
        if state == 6:
            if "stayedStill" not in inpt:
                return 2
            elif "eatOrangeGhost" in inpt:
                return 1
            else:
                return 0
        if state == 2:
            return 2
        return None


# Norm: O^M_score100(westSide | score0)
# Violation specification: F(score0 & !score100 U !westSide)
class TrappedDFA(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, -reward, reset="maintenance")
        self.states = [0, 1, 2]
        self.final = [1]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "score0" in inpt and "westSide" not in inpt:
                return 1
            elif "score0" in inpt and "westSide" in inpt and "scoreGreater100" not in inpt:
                return 2
            else:
                return 0
        if state == 1:
            return 1
        if state == 2:
            if "westSide" not in inpt:
                return 1
            elif "scoreGreater100" not in inpt:
                return 2
            else:
                return 0
        return None


# Norm: O^A_(score100 v lose)(X(eatBlueGhost v eatOrangeGhost) | score0)
# Violation specification: F(score0 & !X(eatBlueGhost | eatOrangeGhost) U (score100 | lose))
class EarlyBirdDFA1(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, -reward, reset="achievement")
        self.states = [0, 1, 2]
        self.final = [2]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "score0" not in inpt:
                return 0
            elif "scoreGreater100" not in inpt and "lose" not in inpt:
                return 1
            else:
                return 2
        if state == 1:
            if "score0" not in inpt and ("eatBlueGhost" in inpt or "eatOrangeGhost" in inpt):
                return 0
            elif (
                "scoreGreater100" not in inpt
                and "lose" not in inpt
                and ("score0" in inpt or "eatBlueGhost" not in inpt)
                and ("score0" in inpt or "eatOrangeGhost" not in inpt)
            ):
                return 1
            else:
                return 2
        if state == 2:
            return 2
        return None


class ScoreHelper(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, -reward, reset="helper")
        self.states = [0, 1, 2, 3, 4]
        self.final = []

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "scoreGreater70" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            if "scoreGreater80" in inpt:
                return 2
            else:
                return 1
        if state == 2:
            if "scoreGreater90" in inpt:
                return 3
            else:
                return 2
        if state == 3:
            if "scoreGreater100" in inpt:
                return 4
            else:
                return 3
        if state == 4:
            return 4
        return None


# Norm: O^A_(score100 v lose)(X(eatPowerPellet) | score0)
# Violation specification: F(score0 & !eatPowerPellet U (score100 | lose))
class EarlyBirdDFA2(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, -reward, reset="achievement")
        self.states = [0, 1, 2]
        self.final = [2]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "score0" in inpt and ("lose" in inpt or "scoreGreater80" in inpt):
                return 2
            elif (
                "score0" in inpt
                and "scoreGreater80" not in inpt
                and "lose" not in inpt
                and "eatPowerPellet" not in inpt
            ):
                return 1
            else:
                return 0
        if state == 1:
            if "lose" in inpt or "scoreGreater80" in inpt:
                return 2
            elif "scoreGreater80" not in inpt and "lose" not in inpt and "eatPowerPellet" not in inpt:
                return 1
            else:
                return 0
        if state == 2:
            return 2
        return None


class EarlyBirdFulfillmentDFA(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, reward, reset="achievement")
        self.states = [0, 1, 2, 3, 4]
        self.final = [3]
        self.violDFA = EarlyBirdDFA1(alph)
        self.reset_on_violation = True

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "score0" not in inpt:
                return 0
            elif "lose" not in inpt and "scoreGreater100" not in inpt:
                return 1
            else:
                return 2
        if state == 1:
            if "score0" not in inpt and ("eatBlueGhost" in inpt or "eatOrangeGhost" in inpt):
                return 3
            elif "scoreGreater100" in inpt or "lose" in inpt:
                return 2
            else:
                return 1
        if state == 2:
            if "eatBlueGhost" not in inpt and "eatOrangeGhost" not in inpt:
                return 4
            elif "score0" not in inpt:
                return 3
            elif "scoreGreater100" in inpt or "lose" in inpt:
                return 2
            else:
                return 1
        if state == 3:
            if "score0" not in inpt:
                return 3
            elif "scoreGreater100" in inpt or "lose" in inpt:
                return 2
            else:
                return 1
        if state == 4:
            return 4
        return None


class EarlyBirdFulfillmentDFA2(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, reward, reset="achievement")
        self.states = [0, 1, 2, 3, 4]
        self.final = [3]
        self.violDFA = EarlyBirdDFA2(alph)
        self.reset_on_violation = True

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "score0" not in inpt:
                return 0
            elif "lose" not in inpt and "scoreGreater80" not in inpt:
                return 1
            else:
                return 2
        if state == 1:
            if "score0" not in inpt and "eatPowerPellet" in inpt:
                return 3
            elif "scoreGreater80" in inpt or "lose" in inpt:
                return 2
            else:
                return 1
        if state == 2:
            if "eatPowerPellet" not in inpt:
                return 4
            elif "score0" not in inpt:
                return 3
            elif "scoreGreater80" in inpt or "lose" in inpt:
                return 2
            else:
                return 1
        if state == 3:
            if "score0" not in inpt:
                return 3
            elif "scoreGreater80" in inpt or "lose" in inpt:
                return 2
            else:
                return 1
        if state == 4:
            return 4
        return None


# O^A_(score100 | lose) (inCorner | score0)
class ErrandDFA1a(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, -reward, reset="achievement")
        self.states = [0, 1, 2]
        self.final = [2]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "score0" in inpt and ("scoreGreater100" in inpt or "lose" in inpt):
                return 2
            elif "score0" in inpt and "inCorner" not in inpt:
                return 1
            else:
                return 0
        elif state == 1:
            if "scoreGreater100" in inpt or "lose" in inpt:
                return 2
            elif "inCorner" not in inpt:
                return 1
            else:
                return 0
        elif state == 2:
            return 2
        return None


# O^A_(score100 | lose) (inCorner | score0)
class ErrandDFA1b(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, reward, reset="achievement")
        self.states = [0, 1, 2, 3]
        self.final = [2]
        self.violDFA = ErrandDFA1a(alph)

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "score0" not in inpt:
                return 0
            elif "inCorner" in inpt:
                return 2
            elif "lose" in inpt or "scoreGreater100" in inpt:
                return 3
            else:
                return 1
        if state == 1:
            if "inCorner" in inpt:
                return 2
            elif "lose" in inpt or "scoreGreater100" in inpt:
                return 3
            else:
                return 1
        if state == 2:
            if "inCorner" in inpt or "score0" not in inpt:
                return 2
            elif "lose" in inpt or "scoreGreater100" in inpt:
                return 3
            else:
                return 1
        if state == 3:
            return 3
        return None


# O^A_(eatBlueGhost | eatOrangeGhost) (inCorner | score0)
class ErrandDFA2a(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, -reward, reset="achievement")
        self.states = [0, 1, 2]
        self.final = [2]

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "score0" in inpt and ("eatBlueGhost" in inpt or "eatOrangeGhost" in inpt or "lose" in inpt):
                return 2
            elif "score0" in inpt and "inCorner" not in inpt:
                return 1
            else:
                return 0
        elif state == 1:
            if "eatBlueGhost" in inpt or "eatOrangeGhost" in inpt or "lose" in inpt:
                return 2
            elif "inCorner" not in inpt:
                return 1
            else:
                return 0
        elif state == 2:
            return 2
        return None


# O^A_(eatBlueGhost | eatOrangeGhost | lose) (inCorner | score0)
class ErrandDFA2b(DFA):
    def __init__(self, alph: Iterable[Label], reward: float = 0) -> None:
        super().__init__(alph, reward, reset="achievement")
        self.states = [0, 1, 2, 3]
        self.final = [2]
        self.violDFA = ErrandDFA2a(alph)
        self.reset_on_violation = True

    def transition(self, inpt: Container[Label], state: State | None = None) -> int | None:
        if state is None:
            state = self.state
        if state == 0:
            if "score0" not in inpt:
                return 0
            elif "inCorner" in inpt:
                return 2
            elif "lose" in inpt or "eatBlueGhost" in inpt or "eatOrangeGhost" in inpt:
                return 3
            else:
                return 1
        if state == 1:
            if "inCorner" in inpt:
                return 2
            elif "lose" in inpt or "eatBlueGhost" in inpt or "eatOrangeGhost" in inpt:
                return 3
            else:
                return 1
        if state == 2:
            if "inCorner" in inpt or "score0" not in inpt:
                return 2
            elif "lose" in inpt or "eatBlueGhost" in inpt or "eatOrangeGhost" in inpt:
                return 3
            else:
                return 1
        if state == 3:
            return 3
        return None


# class EarlyBirdProductDFA(DFA):
#    def __init__(self, alph, reward=0):
#        super().__init__(alph)
#        self.fulfillDFA = EarlyBirdFulfillmentDFA(alph, reward)
#        self.violDFA = EarlyBirdDFA1(alph)
#        self.states = list(itertools.product(self.fulfillDFA.states, self.violDFA.states))
#        self.final = filter(lambda s: s[0] in self.fulfillDFA.final)
#        self.state = (0,0)

#    def transition(self, state, inpt):
#        if state is None:
#            state = self.state
#        newfulfill = self.fulfillDFA.transition(state[0], inpt)
#        newviol = self.violDFA.transition(state[1], inpt)
#        return (newfulfill, newviol)
