from typing import Any

import jax.numpy as jnp
from jax import random
from jax.scipy.stats import multivariate_normal
from jax.typing import ArrayLike

from scenevironment.distribution import Distribution
from scenevironment.environment import ProbabilisticEnv


class JAXProbabilisticEnv(ProbabilisticEnv[ArrayLike, ArrayLike, ArrayLike, float, Any]):
    def split_rng(self) -> Any:
        """
        Split the internal JAX PRNGKey into three independent keys.

        Returns:
            Tuple[random.PRNGKey, random.PRNGKey, random.PRNGKey]: Three independent PRNGKeys.
        """
        self.rng, rng = random.split(self.rng, 2)
        return rng


class GaussianDistribution(Distribution[ArrayLike, Any]):
    def __init__(self, mean: ArrayLike, cov_chol: ArrayLike):
        self.mean = mean
        self.cov_chol = cov_chol

    def sample(self, rng: Any) -> ArrayLike:
        """
        Generate a sample from the Gaussian distribution.

        Args:
            rng (Any): The random number generator.

        Returns:
            ArrayLike: A sample from the distribution.
        """
        return self.mean + self.cov_chol @ random.normal(rng, shape=self.mean.shape)

    def log_prob(self, value: ArrayLike) -> float:
        """
        Compute the log-probability of a given value.

        Args:
            value (ArrayLike): The value to evaluate.

        Returns:
            float: The log-probability of the value.
        """
        return multivariate_normal.logpdf(value, mean=self.mean, cov=self.cov_chol @ self.cov_chol.T)


class LQGEnv(JAXProbabilisticEnv):
    def __init__(self, params: Any, A, B, C, V, W, Q, R):
        super().__init__(params, random.PRNGKey(0))
        self.initial_state = jnp.zeros(2)  # State includes rng_key

        self.A = A
        self.B = B
        self.C = C
        self.V = V  # Process noise covariance (Cholesky factor)
        self.W = W  # Observation noise covariance (Cholesky factor)
        self.Q = Q  # State cost matrix
        self.R = R  # Control cost matrix

    def state_transition_distribution(self, state, action):
        return GaussianDistribution(
            mean=self.A @ state + self.B @ action,
            cov_chol=self.V,
        )

    def observation_distribution(self, state):
        return GaussianDistribution(
            mean=self.C @ state,
            cov_chol=self.W,
        )

    def reward(self, state: ArrayLike, action: ArrayLike) -> float:
        return -(state.T @ self.Q @ state + action.T @ self.R @ action)  # Example: negative squared state as reward


class TrackingTaskEnv(LQGEnv):
    def __init__(self, params: Any):
        A = jnp.eye(2)  # State transition matrix
        B = jnp.array([[0.0], [1.0]])  # Control input matrix
        C = jnp.array([[1.0, -1.0]])  # Observation matrix
        V = jnp.diag(jnp.array([1.0, 0.5]))  # Process noise covariance
        W = jnp.array([[6.0]])  # Observation noise covariance
        Q = jnp.array([[1.0, -1.0], [-1.0, 1.0]])  # State cost matrix
        R = jnp.array([[0.1]])  # Control cost matrix

        super().__init__(params, A, B, C, V, W, Q, R)


if __name__ == "__main__":
    # Example usage
    env = TrackingTaskEnv(params={})

    state = env.initial_state
    action = jnp.array([1.0])

    next_state, observation, reward = env.step(state, action)
    print("Next state:", next_state)
    print("Observation:", observation)
    print("Reward:", reward)
