from dataclasses import replace

import numpy as np
import pytest
import torch

from fog_ridge import Action, Config
from fog_ridge.mappo import (
    MAPPO,
    TEAM_KINDS,
    central_state,
    collect_team,
    decode_team,
    imitate_team,
    pack_team,
)
from fog_ridge.policy import KINDS
from fog_ridge.team import TeamEnv


def take(env, player, kind, **kwargs):
    assert env.next_agent() == player
    env.submit(player, Action(kind, **kwargs))


def pair(env):
    take(env, 0, "request", target=1)
    take(env, 1, "accept", target=0)


def test_snapshot_frozen_at_start_and_measurement_overlap():
    env = TeamEnv(Config(duration=10), 2, heterogeneous=False)
    for i, agent in enumerate(env.agents):
        agent.config = replace(agent.config, latency=2)
        agent.last_direction = np.array([i + 1.0, -i - 1.0])
    before = [a.last_direction.copy() for a in env.agents]
    pair(env)
    assert not env.action_mask(0)["walk"] and not env.action_mask(0)["score"]
    assert env.action_mask(0)["measure"]
    take(env, 0, "measure")
    take(env, 1, "measure")
    assert env.next_agent() == 0 and env.time == 1
    assert all(not any(r.kind == "snapshot" for r in a.records) for a in env.agents)
    take(env, 0, "measure")
    take(env, 1, "measure")
    assert env.next_agent() == 0 and env.time == 2
    for i, agent in enumerate(env.agents):
        record = next(r for r in agent.records if r.kind == "snapshot")
        np.testing.assert_array_equal(record.direction, before[1 - i])
        assert record.time == 0 and record.value is None and record.source == 1 - i
    assert env.stats["sessions_completed"] == 1
    assert env.stats["overlap_measurements"] == 4


def test_session_end_mid_measurement_preserves_both_records():
    env = TeamEnv(Config(duration=10), 2, heterogeneous=False)
    pair(env)  # L=1, measurement B=2 ends at t=2.
    take(env, 0, "measure", batch=2)
    take(env, 1, "measure", batch=2)
    assert env.next_agent() == 0 and env.time == 1
    assert all(len(a.records) == 1 and a.records[0].kind == "snapshot" for a in env.agents)
    take(env, 0, "continue")
    take(env, 1, "continue")
    assert env.next_agent() == 0 and env.time == 2
    assert all([r.kind for r in a.records] == ["snapshot", "measure"] for a in env.agents)
    assert env.stats["overlap_measurements"] == 2


def test_score_preemption_request_arrives_while_busy():
    env = TeamEnv(Config(duration=10), 2, heterogeneous=False)
    take(env, 0, "score")
    take(env, 1, "request", target=0)
    assert 0 in env.futures
    take(env, 0, "accept", target=1)
    assert 0 not in env.futures and env.stats["score_cancelled"] == 1
    assert env.time == 0 and env.session(0).start == 0
    assert not any(r.kind == "score" for r in env.agents[0].records)


def test_walk_finishes_before_reserved_session_begins():
    env = TeamEnv(Config(duration=10), 3, heterogeneous=False)
    position = env.agents[0].position.copy()
    take(env, 0, "walk", displacement=[1, 0])
    take(env, 1, "request", target=0)
    take(env, 2, "wait", seconds=0.05)
    take(env, 0, "accept", target=1)
    assert env.session(0).start is None
    assert not env.action_mask(1)["walk"]
    take(env, 1, "measure")
    assert env.next_agent() == 2 and env.time == 0.05  # Nonparticipants keep moving.
    take(env, 2, "wait", seconds=1)
    assert env.next_agent() == 0 and env.time == 0.1
    assert env.session(0).start == 0.1
    np.testing.assert_array_equal(
        next(r for r in env.session(0).snapshots if r.source == 0).position, position + [1, 0]
    )


def test_immediate_403_does_not_reveal_busy_in_target_mask():
    env = TeamEnv(Config(duration=10), 3, heterogeneous=False)
    pair(env)
    assert env.target_mask(2, "request")[0]
    take(env, 2, "request", target=0)
    assert env.time == 0 and env.stats["403"] == 1
    assert env.last_result[2] == "403" and env.outgoing(2) is None


def test_competing_requests_accept_one_invalidates_rest():
    env = TeamEnv(Config(duration=10), 3, heterogeneous=False)
    take(env, 0, "request", target=1)
    take(env, 1, "measure", batch=2)
    take(env, 2, "request", target=1)
    take(env, 0, "wait")
    take(env, 1, "accept", target=0)
    assert env.links[0] == env.links[1] and 2 not in env.links
    assert not env.requests and env.last_result[2] == "403"


def test_ignore_keeps_work_running_and_request_expires():
    env = TeamEnv(Config(duration=10), 2, heterogeneous=False)
    take(env, 0, "request", target=1)
    take(env, 1, "ignore", target=0)
    assert not env.action_mask(1)["ignore"] and env.action_mask(1)["accept"]
    take(env, 0, "wait", seconds=6)
    take(env, 1, "measure", batch=8)
    assert env.next_agent() == 0 and env.time == 5
    assert not env.requests and env.stats["expired"] == 1
    assert 1 in env.futures and not env.agents[1].records
    assert env.action_mask(1)["continue"]


def test_reject_and_disconnect_deliver_nothing():
    env = TeamEnv(Config(duration=10), 2, heterogeneous=False)
    take(env, 0, "request", target=1)
    take(env, 1, "reject", target=0)
    assert not env.requests and env.last_result[0] == "rejected"
    pair(env)
    take(env, 0, "measure", batch=2)
    take(env, 1, "disconnect")
    assert not env.links and not env.sessions and 0 in env.futures
    assert env.stats["disconnected"] == 1 and env.stats["snapshots"] == 0
    assert not any(a.records for a in env.agents)


def test_snapshot_overflow_during_measurement_does_not_overwrite():
    env = TeamEnv(Config(duration=10, notebook_size=1), 2, heterogeneous=False)
    pair(env)
    take(env, 0, "measure", batch=2)
    take(env, 1, "measure", batch=2)
    assert env.next_agent() == 0 and env.time == 1
    take(env, 0, "continue")
    take(env, 1, "continue")
    assert env.next_agent() == 0 and env.time == 2
    assert env.agents[0].pending.kind == "measure"
    take(env, 0, "delete", slots=(0,))
    assert env.agents[0].pending is None and env.agents[0].records[0].kind == "measure"


@pytest.mark.parametrize("latency,delivered", [(2, False), (1, True)])
def test_session_crossing_terminal_time(latency, delivered):
    env = TeamEnv(Config(duration=1, latency=latency), 2, heterogeneous=False)
    pair(env)
    take(env, 0, "wait")
    take(env, 1, "wait")
    assert env.next_agent() is None and env.done
    assert env.stats["snapshots"] == (2 if delivered else 0)


def test_communication_action_likelihood_replay_and_target_gradient():
    torch.manual_seed(5)
    env = TeamEnv(players=3)
    model = MAPPO(width=16)
    data = pack_team(env, 0, 20)
    action = torch.zeros(1, 9)
    action[0, 0] = TEAM_KINDS.index("request")
    action[0, 8] = 1
    _, before, _ = model.act(data, action)
    _, replay, _ = model.act(data, action)
    torch.testing.assert_close(before, replay)
    (-before.mean()).backward()
    assert model.partner_query.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.partner_query.weight.grad).all()


def test_no_communication_ablation_and_teacher_produces_exchange():
    torch.manual_seed(10)
    model = MAPPO(capacity=3, width=16)
    config = Config(duration=12, notebook_size=3)
    optimizer = torch.optim.Adam(model.parameters())
    stats = imitate_team(model, optimizer, config, [1], 2, "min", "cpu", epochs=1)
    assert stats["requests"] > 0 and stats["sessions_completed"] > 0 and stats["snapshots"] > 0
    _, reports = collect_team(model, config, [1], players=2, communication=False)
    assert reports[0]["communication"]["requests"] == 0


def test_simultaneous_measurement_and_snapshot_queue_requires_two_evictions():
    from fog_ridge.simulator import Record

    env = TeamEnv(Config(duration=10, notebook_size=1), 2, heterogeneous=False)
    for agent in env.agents:
        agent.records.append(
            Record("measure", agent.position.copy(), 0, direction=np.ones(2), value=1, batch=1)
        )
    pair(env)
    take(env, 0, "measure")
    take(env, 1, "measure")
    assert env.next_agent() == 0 and env.time == 1
    assert env.agents[0].pending.kind == "measure"
    assert len(env.deliveries[0]) == 1
    take(env, 0, "delete", slots=(0,))
    assert env.agents[0].records[0].kind == "measure"
    assert env.agents[0].pending.kind == "snapshot"
    take(env, 1, "delete", slots=(0,))
    take(env, 0, "delete", slots=(0,))
    assert env.agents[0].records[0].kind == "snapshot"
    assert env.agents[0].pending is None and not env.deliveries[0]


def test_departure_archives_position_and_makes_it_a_walk_target():
    env = TeamEnv(Config(duration=10, notebook_size=2), 2, heterogeneous=False)
    env.agents[1].position += np.array([2.0, -1.0])
    archived_position = env.agents[1].position.copy()
    env._depart(1)

    assert not env.alive[1] and env.stats["departures"] == 1
    np.testing.assert_array_equal(env.archives[0].position, archived_position)
    assert not env.target_mask(0, "request")[1]

    model = MAPPO(capacity=4, batch_max=8, width=16)
    data = pack_team(env, 0, model.settings["capacity"])
    assert data["valid"][0, 0] and not data["manageable"][0, 0]
    assert data["legal"][0, KINDS.index("walk")]
    sample = torch.zeros(9)
    sample[0] = TEAM_KINDS.index("walk")
    sample[3] = 0  # The archive is the first virtual position token.
    action = decode_team(env, 0, sample)
    np.testing.assert_allclose(action.displacement, [2.0, -1.0])

    visible_state = central_state(env, 0)
    env.archives_visible = False
    hidden = pack_team(env, 0, model.settings["capacity"])
    assert not hidden["valid"].any()
    assert not hidden["legal"][0, KINDS.index("walk")]
    assert not torch.equal(visible_state, central_state(env, 0))


@pytest.mark.parametrize("seed", range(8))
def test_random_protocol_actions_preserve_invariants(seed):
    env = TeamEnv(Config(duration=8, notebook_size=2, dimension=8), 4, seed=seed)
    rng = np.random.default_rng(seed)
    initial = env.potential
    for _ in range(20_000):
        player = env.next_agent()
        if player is None:
            break
        local = env.agents[player]
        kinds = [kind for kind, allowed in env.action_mask(player).items() if allowed]
        kind = str(rng.choice(kinds))
        kwargs = {}
        if kind in ("request", "accept", "reject", "ignore"):
            kwargs["target"] = int(rng.choice(np.flatnonzero(env.target_mask(player, kind))))
        if kind == "delete":
            kwargs["slots"] = (int(rng.integers(len(local.records))),)
        if kind == "summarize":
            indices = [i for i, record in enumerate(local.records) if record.direction is not None]
            kwargs["slots"] = tuple(int(i) for i in rng.choice(indices, 2, replace=False))
        if kind == "walk":
            kwargs["displacement"] = rng.normal(size=8)
        if kind == "measure":
            kwargs["batch"] = int(rng.integers(1, 9))
        env.submit(player, Action(kind, **kwargs))
        assert all(len(a.records) <= 2 for a in env.agents)
        assert all(np.isfinite(a.position).all() for a in env.agents)
        for session in env.sessions.values():
            assert env.links[session.a] == env.links[session.b]
        for request in env.requests.values():
            assert request.sender != request.target and request.expires > env.time
            assert request.sender not in env.links and request.target not in env.links
    else:
        pytest.fail("zero-time protocol loop")
    assert env.done and not env.links and not env.requests and not env.sessions
    assert sum(row.get("team_reward", 0) for row in env.trace) == pytest.approx(
        env.potential - initial
    )


def test_received_direction_can_drive_a_later_walk():
    from fog_ridge.mappo import decode_team

    env = TeamEnv(Config(duration=10), 2, heterogeneous=False)
    env.agents[1].last_direction = np.array([0.0, 1.0])
    position = env.agents[0].position.copy()
    pair(env)
    take(env, 0, "wait")
    take(env, 1, "wait")
    assert env.next_agent() == 0 and env.time == 1
    np.testing.assert_array_equal(env.agents[0].position, position)
    sample = torch.tensor([2, 0, 0, 0, 0, 0, np.log(0.1), -12, 0])
    env.submit(0, decode_team(env, 0, sample))
    take(env, 1, "wait")
    assert env.next_agent() == 0 and env.time == pytest.approx(1.1)
    np.testing.assert_allclose(env.agents[0].position - position, [0, 0.1], atol=1e-7)


def test_stochastic_evaluation_is_reproducible_and_preserves_training_rng():
    from fog_ridge.mappo import evaluate_team

    model = MAPPO(width=16)
    config = Config(duration=3)
    state = torch.get_rng_state()
    first = evaluate_team(model, config, [7], 2, "min")
    torch.testing.assert_close(state, torch.get_rng_state())
    second = evaluate_team(model, config, [7], 2, "min")
    assert first == second and not first["deterministic"]


def test_other_players_busy_status_is_private():
    env = TeamEnv(Config(duration=5), 3, heterogeneous=False)
    take(env, 0, "wait")
    before = pack_team(env, 0, 20)
    take(env, 1, "request", target=2)
    take(env, 2, "accept", target=1)
    after = pack_team(env, 0, 20)
    for key in before:
        torch.testing.assert_close(before[key], after[key])


def test_heterogeneous_latency_waits_for_the_slower_endpoint():
    env = TeamEnv(Config(duration=5), 2, heterogeneous=False)
    env.agents[0].config = replace(env.agents[0].config, latency=0.25)
    env.agents[1].config = replace(env.agents[1].config, latency=3)
    pair(env)
    take(env, 0, "wait", seconds=2)
    take(env, 1, "wait", seconds=2)
    assert env.next_agent() == 0 and env.time == 2
    assert env.stats["snapshots"] == 0
    take(env, 0, "wait", seconds=2)
    take(env, 1, "wait", seconds=2)
    assert env.next_agent() == 0 and env.time == 3
    assert env.stats["snapshots"] == 2
