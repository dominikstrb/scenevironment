from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from jax import random
from jax.scipy.stats import multivariate_normal
from jax.typing import ArrayLike

T = TypeVar("T")  # Generic type for samples
RNG = TypeVar("RNG")  # Generic type for the random number generator


class Distribution(ABC, Generic[T, RNG]):
    @abstractmethod
    def sample(self, rng: RNG) -> T:
        """
        Generate a sample from the distribution.

        Args:
            rng (RNG): The random number generator.

        Returns:
            T: A sample from the distribution.
        """
        pass

    @abstractmethod
    def log_prob(self, value: T) -> T:
        """
        Compute the log-probability of a given value.

        Args:
            value (T): The value to evaluate.

        Returns:
            float: The log-probability of the value.
        """
        pass


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
