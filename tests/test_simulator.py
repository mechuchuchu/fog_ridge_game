import numpy as np
import pytest

from fog_ridge import Action, Config, SoloEnv
from fog_ridge.baseline import DecayingSGD, rollout
from fog_ridge.landscape import Landscape


@pytest.mark.parametrize("dimension", [2, 16, 128, 512])
def test_closed_form_matches_direct_ensemble(dimension):
    rng = np.random.default_rng(5)
    world = Landscape.generate(rng, dimension, 100, 2)
    x = rng.normal(size=dimension)
    assert world.score(x) == pytest.approx(world.values(x).mean())
    value, direction = world.measure(x, np.arange(100))
    assert value == pytest.approx(world.score(x))
    np.testing.assert_allclose(direction, -world.curvature * (x - world.center))
    h = 1e-5
    probe = rng.normal(size=dimension)
    finite_diff = (world.score(x + h * probe) - world.score(x - h * probe)) / (2 * h)
    assert finite_diff == pytest.approx(-direction @ probe, rel=1e-7)


def test_same_map_remeasurement_and_score_privacy():
    env = SoloEnv()
    env.step(Action("measure"))
    env.step(Action("measure"))
    np.testing.assert_array_equal(env.records[0].direction, env.records[1].direction)
    assert env.records[0].value == env.records[1].value
    assert env.observe()["dashboard"] is None
    assert "gap" not in env.observe()
    env.step(Action("score"))
    assert env.dashboard == env.records[1].value
    assert env.records[-1].value == pytest.approx(env.landscape.values(env.position).mean())


def test_current_map_in_batch_and_unique_samples():
    env = SoloEnv(Config(batch_max=100))
    env.step(Action("measure", batch=100))
    np.testing.assert_allclose(
        env.records[-1].direction, -env.landscape.curvature * (env.position - env.landscape.center)
    )
    assert env.records[-1].value == pytest.approx(env.landscape.score(env.position))


def test_dashboard_completion_order():
    env = SoloEnv(Config(measurement_time=5))
    env.step(Action("measure"))
    assert env.dashboard == env.last_measurement
    assert env.dashboard_time == 5
    env.step(Action("measure", batch=2))
    assert env.dashboard_time == 15
    assert env.dashboard == env.last_measurement


@pytest.mark.parametrize("duration,completed", [(0.05, False), (0.1, True)])
def test_terminal_walk_and_clipping(duration, completed):
    env = SoloEnv(Config(duration=duration, step_max=0.5))
    initial = env.position.copy()
    _, _, done, info = env.step(Action("walk", displacement=np.array([100.0, 0])))
    assert done and info["completed"] == completed
    assert np.linalg.norm(env.position - initial) == pytest.approx(0.5 if completed else 0)


def test_incomplete_measurement_has_no_result():
    env = SoloEnv(Config(duration=0.5))
    env.step(Action("measure"))
    assert env.done and env.records == [] and env.last_measurement is None


def test_overflow_forces_management_and_summary():
    env = SoloEnv(Config(notebook_size=2))
    for _ in range(3):
        env.step(Action("measure"))
    assert env.pending is not None
    with pytest.raises(ValueError):
        env.step(Action("walk", displacement=np.zeros(2)))
    time = env.time
    env.step(Action("summarize", slots=(0, 1)))
    assert env.time == time and len(env.records) == 2 and env.pending is None
    assert env.records[0].kind == "summary"


def test_invalid_action_is_atomic():
    env = SoloEnv()
    for action in [
        Action("measure", batch=0),
        Action("walk", displacement=[np.nan, 0]),
        Action("wait", seconds=0),
        Action("delete", slots=(0,)),
    ]:
        with pytest.raises(ValueError):
            env.step(action)
        assert env.time == 0 and env.trace == []


def test_seeded_rollout_and_telescoping_reward():
    config = Config(dimension=16, duration=60)
    a, b = SoloEnv(config, 12), SoloEnv(config, 12)
    initial = a.potential
    result = rollout(a, DecayingSGD())
    assert result == rollout(b, DecayingSGD())
    assert a.trace == b.trace
    assert result["reward"] == pytest.approx(a.potential - initial)
    assert result["final_gap"] < result["initial_gap"]


def test_rotation_invariant_observation_and_decoder():
    env = SoloEnv()
    env.step(Action("measure"))
    env.step(Action("walk", displacement=[1, 2]))
    env.step(Action("measure"))
    before = env.observe()
    weights = np.array([0.3, 0.7])
    walk = env.compose_walk(weights, weights).displacement
    rotation = np.array([[0, -1], [1, 0]])
    env.position = rotation @ env.position
    for record in env.records:
        record.position = rotation @ record.position
        record.direction = rotation @ record.direction
    after = env.observe()
    for key in ("records", "direction_cosines", "direction_offset_cosines"):
        np.testing.assert_allclose(before[key], after[key], atol=1e-12)
    np.testing.assert_allclose(env.compose_walk(weights, weights).displacement, rotation @ walk)


@pytest.mark.parametrize("seed", range(20))
def test_random_actions_preserve_invariants(seed):
    env = SoloEnv(Config(duration=30, notebook_size=3, dimension=8), seed)
    rng = np.random.default_rng(seed)
    rewards = 0.0
    initial = env.potential
    while not env.done:
        previous = env.time
        legal = [k for k, valid in env.action_mask().items() if valid and k != "summarize"]
        kind = str(rng.choice(legal))
        action = Action(
            kind,
            batch=int(rng.integers(1, 9)),
            displacement=rng.normal(size=8),
            seconds=float(rng.uniform(0.1, 4)),
            slots=(0,),
        )
        _, reward, _, _ = env.step(action)
        rewards += reward
        assert previous <= env.time <= env.config.duration
        assert len(env.records) <= 3
        assert np.isfinite(env.position).all()
        assert 0 <= env.current_map < 100
    assert rewards == pytest.approx(env.potential - initial)
    assert not any(env.action_mask().values())
