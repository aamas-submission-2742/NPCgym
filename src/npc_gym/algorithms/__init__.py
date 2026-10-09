"""Learning algorithms provided by NPC Gym."""

from npc_gym.algorithms.tabular_often import TabularOFTEN, TeachingSession, TeachingStats, TeachingStep
from npc_gym.algorithms.tabular_q_learning import TabularQLearning as TabularQLearning
from npc_gym.algorithms.tabular_q_learning import TrainingCallback as TrainingCallback
from npc_gym.algorithms.tabular_q_learning import TrainingStep as TrainingStep

__all__ = [
    "TabularOFTEN",
    "TabularQLearning",
    "TeachingSession",
    "TeachingStats",
    "TeachingStep",
    "TrainingCallback",
    "TrainingStep",
]
