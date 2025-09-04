from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from scenevironment.distribution import RNG
from scenevironment.environment import Action, Observation, ProbabilisticEnv

InternalState = TypeVar("InternalState")


class Agent(ABC, Generic[InternalState, Action]):
    @abstractmethod
    def behave(self, internal_state: InternalState, rng: RNG) -> Action:
        """
        Select an action given the current state.
        """
        pass

    @abstractmethod
    def update_state(self, internal_state: InternalState, action: Action, obs: Observation) -> InternalState:
        """
        Update the internal state of the agent.
        """
        pass


class RationalAgent(Agent[InternalState, Action], Generic[InternalState, Action]):
    def __init__(self, internal_model: ProbabilisticEnv):
        self.internal_model = internal_model

        self.policy = self.internal_model.optimal_policy()
        self.bayesian_belief_update = self.internal_model.bayesian_belief_update()

    def behave(self, internal_state: InternalState, rng: RNG = None) -> Action:
        return self.policy(internal_state)

    def update_state(self, internal_state, action, obs):
        return self.bayesian_belief_update(internal_state, action, obs)
