from typing import Any, Callable

import jax
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

    def optimal_policy(self) -> Callable:
        K = self.lqr_gains()

        def _policy(state):
            return -K @ state

        return _policy

    def bayesian_belief_update(self) -> Callable:
        K = self.kalman_gains()

        def _belief_update(state, action, obs):
            return (self.A @ state + self.B @ action) + K @ (obs - self.C @ (self.A @ state + self.B @ action))

        return _belief_update

    def lqr_gains(self) -> ArrayLike:
        """
        Compute the optimal LQR policy

        Returns:
            Callable: A function that takes the current state and returns the optimal action.
        """

        X = dare_sda_solver(self.A, self.B, self.Q, self.R)
        # Inverse term: (R + B^T @ X @ B)^-1
        inv_term = jnp.linalg.inv(self.R + self.B.T @ X @ self.B)

        # Gain term: B^T @ X @ A
        gain_term = self.B.T @ X @ self.A

        # Full gain matrix K
        K = inv_term @ gain_term

        return K

    def kalman_gains(self) -> ArrayLike:
        Sigma = dare_sda_solver(self.A.T, self.C.T, self.V @ self.V.T, self.W @ self.W.T)

        # 2. Compute the Kalman gain K
        # K = A * Sigma * H' * (H * Sigma * H' + V)^-1
        K = (self.A @ Sigma @ self.C.T) @ jnp.linalg.inv(self.C @ Sigma @ self.C.T + self.W @ self.W.T)
        return K


def dare_sda_solver(A, B, Q, R, S=None, num_iterations=10):
    """
    Solves the discrete algebraic Riccati equation using the
    Structured Doubling Algorithm (SDA).

    The DARE is of the form A'XA - X - (A'XB + S)(R + B'XB)^-1(B'XA + S') + Q = 0.
    This simplified version assumes S=0.
    """
    n, _ = A.shape
    if S is None:
        S = jnp.zeros((n, B.shape[1]))

    # Define the update function for one iteration
    def sda_iteration(carry, x):
        Ak, Gk, Hk = carry

        # Intermediate inverse terms
        inv_I_plus_Gk_Hk = jnp.linalg.inv(jnp.eye(n) + Gk @ Hk)
        inv_I_plus_Hk_Gk = jnp.linalg.inv(jnp.eye(n) + Hk @ Gk)

        # SDA updates
        Ak_next = Ak @ inv_I_plus_Gk_Hk @ Ak
        Gk_next = Gk + Ak @ inv_I_plus_Gk_Hk @ Gk @ Ak.T
        Hk_next = Hk + Ak.T @ inv_I_plus_Hk_Gk @ Hk @ Ak

        return (Ak_next, Gk_next, Hk_next), None

    # Initial matrices for SDA
    A0 = A
    G0 = B @ jax.scipy.linalg.solve(R, B.T)
    H0 = Q

    # JIT-compile the iterative loop
    (Af, Gf, Hf), _ = jax.lax.scan(sda_iteration, (A0, G0, H0), jnp.arange(num_iterations))

    return Hf


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
