"""Gymnasium adapter to the SAME environment used by Baseten.

Action: seven normalized complete-toss parameters, not two raw torques.
Observation: 23 normalized values. reset -> (obs, info); step -> five values.
"""
from common.config import settings
from toss.environment import TrainableTossEnv
class PancakeEnv(TrainableTossEnv):
    def __init__(self,record=False):
        super().__init__(settings(),profile='hardware',record=record)
