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


if __name__ == "__main__":
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    from scenevironment.lqg import TrackingTaskEnv

    # Example usage
    env = TrackingTaskEnv(params={})
    state = env.initial_state

    agent = RationalAgent(internal_model=env)
    internal_state = state  # Initial internal state

    states = []
    for _ in range(1000):
        # compute the action based on current internal state
        action = agent.behave(internal_state)

        # take a step in the environment
        state, observation, reward = env.step(state, action)
        states.append(state)

        # update the internal state based on agent's action and observation
        internal_state = agent.update_state(internal_state, action, observation)

    x = jnp.stack(states)
    plt.plot(x[:, 0])
    plt.plot(x[:, 1])
    plt.show()
