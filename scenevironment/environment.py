from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from scenevironment.distribution import RNG, Distribution

State = TypeVar("State")
Action = TypeVar("Action")
Observation = TypeVar("Observation")
Reward = TypeVar("Reward", bound=float)


class Env(ABC, Generic[State, Action, Observation, Reward]):
    def __init__(self, params: Any):
        self.params = params

    @abstractmethod
    def step(self, state: State, action: Action) -> tuple[State, Observation, Reward]:
        pass

    @abstractmethod
    def obs(self, state: State) -> Observation:
        pass

    @abstractmethod
    def reward(self, state: State, action: Action) -> Reward:
        pass


class ProbabilisticEnv(
    Env[State, Action, Observation, Reward],
    Generic[State, Action, Observation, Reward, RNG],
):
    def __init__(self, params: Any, rng: RNG):
        super().__init__(params)
        self.rng = rng  # Internal RNG instance

    def step(self, state: State, action: Action) -> tuple[State, Observation, Reward]:
        """
        Perform a step in the environment based on probability distributions.

        Args:
            state (State): The current state.
            action (Action): The action to take.

        Returns:
            Tuple[State, Observation, Reward]: The next state, observation, and reward.
        """
        # Split the RNG for state, observation, and reward
        rng_state = self.split_rng()

        # Sample from the distributions
        state_dist = self.state_transition_distribution(state, action)
        next_state = state_dist.sample(rng_state)

        return next_state, self.obs(state), self.reward(state, action)

    def obs(self, state: State) -> Observation:
        rng_obs = self.split_rng()

        return self.observation_distribution(state).sample(rng_obs)

    def split_rng(self) -> RNG:
        """
        Split the internal RNG into three independent RNGs.

        Returns:
            Tuple[RNG, RNG, RNG]: Three independent RNGs.
        """
        raise NotImplementedError("Subclasses must implement RNG splitting.")

    def state_transition_distribution(self, state: State, action: Action) -> Distribution[State, RNG]:
        raise NotImplementedError

    def observation_distribution(self, state: State) -> Distribution[Observation, RNG]:
        raise NotImplementedError
