from dataclasses import replace

import numpy as np
import pytest
import torch

from fog_ridge import Action, Config
from fog_ridge.mappo import (
    MAPPO,
    central_state,
    collect_team,
    load_mappo,
    pack_team,
    save_mappo,
)
from fog_ridge.ppo import update
from fog_ridge.team import TeamEnv

torch.set_num_threads(1)


def test_team_shared_world_independent_sampling_and_atomic_completion():
    env = TeamEnv(Config(duration=1), 2, seed=4, heterogeneous=False)
    a, b = env.agents
    np.testing.assert_array_equal(a.landscape.centers, b.landscape.centers)
    np.testing.assert_array_equal(a.position, b.position)
    before = a.position.copy()
    assert env.next_agent() == 0
    env.submit(0, Action("walk", displacement=[1, 0]))
    np.testing.assert_array_equal(env.agents[0].position, before)
    assert env.next_agent() == 1
    env.submit(1, Action("walk", displacement=[0, 1]))
    assert env.next_agent() == 0
    assert env.time == pytest.approx(0.1)
    np.testing.assert_array_equal(env.agents[0].position, before + [1, 0])
    np.testing.assert_array_equal(env.agents[1].position, before + [0, 1])
    assert len([row for row in env.trace if "team_reward" in row]) == 1


def test_asynchronous_events_do_not_commit_future_results():
    env = TeamEnv(Config(duration=10), 2, heterogeneous=False)
    env.agents[1].config = replace(env.config, measurement_time=2)
    env.submit(0, Action("measure"))
    env.submit(1, Action("measure"))
    assert env.next_agent() == 0 and env.time == 1
    assert len(env.agents[0].records) == 1 and len(env.agents[1].records) == 0
    assert env.agents[0].dashboard is None and env.agents[1].last_measurement is None
    env.submit(0, Action("wait", seconds=4))
    assert env.next_agent() == 1 and env.time == 2
    env.submit(1, Action("wait", seconds=3))
    assert env.next_agent() == 0 and env.time == 5
    assert all(a.dashboard == a.last_measurement for a in env.agents)


def test_terminal_actions_and_invalid_submission():
    env = TeamEnv(Config(duration=0.05), 2, heterogeneous=False)
    initial = env.gap
    with pytest.raises(ValueError):
        env.submit(1, Action("wait"))
    with pytest.raises(ValueError):
        env.submit(0, Action("measure", batch=0))
    assert list(env.ready) == [0, 1] and not env.futures
    env.submit(0, Action("walk", displacement=[1, 0]))
    env.submit(1, Action("walk", displacement=[1, 0]))
    assert env.next_agent() is None and env.done
    assert env.time == 0.05 and env.gap == initial


@pytest.mark.parametrize("aggregation", ["min", "mean"])
def test_agent_intervals_receive_full_team_reward(aggregation):
    torch.manual_seed(4)
    config = Config(duration=6, notebook_size=3)
    model = MAPPO(capacity=3, width=16)
    batch, reports = collect_team(
        model, config, [1, 2], players=3, aggregation=aggregation, gamma=1, lam=1
    )
    for report in reports:
        np.testing.assert_allclose(report["agent_rewards"], report["potential_change"], atol=1e-8)
        assert len(report["decisions"]) == 3 and min(report["decisions"]) > 0
    assert batch["critic"].shape[1:] == (4, 15)
    assert batch["data"]["players"].shape[1:] == (3, 11)
    _, logps, _ = model.act(batch["data"], batch["actions"])
    torch.testing.assert_close(logps, batch["logp"], atol=1e-5, rtol=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.critic.head[-1].weight.detach().clone()
    metrics = update(model, optimizer, batch, epochs=1, minibatch=16)
    assert all(np.isfinite(v) for v in metrics.values())
    assert not torch.equal(before, model.critic.head[-1].weight)


def test_public_observation_excludes_private_teammate_state():
    env = TeamEnv(players=2)
    model = MAPPO(width=16)
    before = pack_team(env, 0, 20)
    central_before = central_state(env, 0)
    env.agents[1].position += 10
    after = pack_team(env, 0, 20)
    for key in before:
        torch.testing.assert_close(before[key], after[key])
    assert not torch.equal(central_before, central_state(env, 0))
    with torch.no_grad():
        for parameter in model.critic.parameters():
            parameter.add_(10)
    first = model.act(before, deterministic=True)[0]
    second = model.act(after, deterministic=True)[0]
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("players", [1, 2, 6])
def test_no_agent_id_and_variable_player_count(players):
    env = TeamEnv(players=players)
    model = MAPPO(width=16)
    data = pack_team(env, 0, 20)
    state = central_state(env, 0)
    action, logp, _ = model.act(data, deterministic=True)
    value = model.value(state)
    permutation = torch.randperm(players)
    data["players"] = data["players"][:, permutation]
    data["targets"] = data["targets"][:, :, permutation]
    state = torch.cat([state[:, :1], state[:, 1:][:, permutation]], 1)
    after_action = model.act(data, deterministic=True)[0]
    torch.testing.assert_close(action[:, :8], after_action[:, :8])
    if 6 <= int(action[0, 0]) < 10:
        action[:, 8] = (permutation == int(action[0, 8])).nonzero()[0, 0]
    torch.testing.assert_close(action, after_action)
    torch.testing.assert_close(logp, model.act(data, action)[1])
    torch.testing.assert_close(value, model.value(state))


def test_team_checkpoint_resume_reproduces_next_update(tmp_path):
    torch.manual_seed(9)
    config = Config(duration=3, notebook_size=2)
    model = MAPPO(capacity=2, width=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch, _ = collect_team(model, config, [0], players=2)
    update(model, optimizer, batch, epochs=1)
    path = tmp_path / "mappo.pt"
    save_mappo(path, model, optimizer, config, 1, 1, {"players": 2, "aggregation": "mean"})
    restored, restored_config, checkpoint = load_mappo(path)
    assert restored_config == config and checkpoint["training"]["players"] == 2
    resumed_optimizer = torch.optim.Adam(restored.parameters())
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng"])
    batch, reports = collect_team(model, config, [1], players=2, aggregation="mean")
    update(model, optimizer, batch, epochs=1)
    torch.set_rng_state(checkpoint["torch_rng"])
    next_batch, next_reports = collect_team(restored, config, [1], players=2, aggregation="mean")
    update(restored, resumed_optimizer, next_batch, epochs=1)
    assert reports == next_reports
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b)
