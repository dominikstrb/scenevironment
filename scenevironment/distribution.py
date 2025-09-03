from abc import ABC, abstractmethod
from typing import Generic, TypeVar

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
    def log_prob(self, value: T) -> float:
        """
        Compute the log-probability of a given value.

        Args:
            value (T): The value to evaluate.

        Returns:
            float: The log-probability of the value.
        """
        pass
