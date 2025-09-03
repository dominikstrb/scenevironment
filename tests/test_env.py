import jax.numpy as jnp

from scenevironment.lqg import TrackingTaskEnv


def test_foo():
    env = TrackingTaskEnv(params={})

    state = env.initial_state
    action = jnp.array([1.0])

    next_state, observation, reward = env.step(state, action)

    assert next_state.shape == (2,)
