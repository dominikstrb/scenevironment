import jax.numpy as jnp

from scenevironment.agent import RationalAgent
from scenevironment.lqg import TrackingTaskEnv


def test_lqg_env():
    env = TrackingTaskEnv(params={})

    state = env.initial_state
    action = jnp.array([1.0])

    next_state, observation, reward = env.step(state, action)

    assert next_state.shape == (2,)


def test_agent():
    env = TrackingTaskEnv(params={})
    state = env.initial_state

    agent = RationalAgent(internal_model=env)
    internal_state = state  # Initial internal state

    action = agent.behave(internal_state)

    state, observation, reward = env.step(state, action)

    internal_state = agent.update_state(internal_state, action, observation)

    assert action.shape == (1,)
    assert internal_state.shape == (2,)
