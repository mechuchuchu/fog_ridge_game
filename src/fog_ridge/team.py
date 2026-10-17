"""Continuous-time team simulator with handshakes and overlapping communication."""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from .simulator import Config, Record, SoloEnv

COMM_KINDS = ("request", "accept", "reject", "ignore", "disconnect", "continue")


@dataclass
class Request:
    sender: int
    target: int
    expires: float
    ignored: bool = False


@dataclass
class Session:
    a: int
    b: int
    start: float | None = None
    end: float = float("inf")
    snapshots: tuple | None = None


class TeamEnv:
    """Local effects commit at completion; communication never exposes future copies."""

    def __init__(
        self,
        config=None,
        players=4,
        seed=0,
        aggregation="min",
        heterogeneous=True,
        communication=True,
        request_timeout=5.0,
    ):
        config = Config() if config is None else config
        if not isinstance(players, int) or players < 1:
            raise ValueError("players must be a positive integer")
        if aggregation not in ("min", "mean"):
            raise ValueError("aggregation must be min or mean")
        if not np.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        self.config, self.aggregation = config, aggregation
        self.communication, self.request_timeout = communication, request_timeout
        self.time, self.done = 0.0, False
        self.agents = [SoloEnv(config, seed) for _ in range(players)]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 8191]))
        self.futures, self.requests, self.sessions, self.links = {}, {}, {}, {}
        self.deliveries = [deque() for _ in self.agents]
        self.ready = deque(range(players))
        self.trace = []
        self.last_result = ["" for _ in self.agents]
        self.stats = dict.fromkeys(
            (
                "requests",
                "403",
                "accepted",
                "rejected",
                "expired",
                "sessions_started",
                "sessions_completed",
                "disconnected",
                "score_cancelled",
                "snapshots",
                "overlap_measurements",
            ),
            0,
        )
        self.next_session = 0
        for agent in self.agents:
            if heterogeneous:
                agent.config = replace(
                    config,
                    measurement_time=float(np.exp(rng.uniform(np.log(0.5), np.log(2)))),
                    latency=float(np.exp(rng.uniform(np.log(0.25), np.log(5)))),
                )
            agent.rng = np.random.default_rng(rng.integers(0, 2**63))
            agent.current_map = int(agent.rng.integers(100))

    @property
    def gaps(self):
        return np.array([a.landscape.gap(a.position) for a in self.agents])

    @property
    def gap(self):
        return float(self.gaps.min() if self.aggregation == "min" else self.gaps.mean())

    @property
    def potential(self):
        return float(-np.log10(self.gap + 1e-12))

    def incoming(self, player):
        return [r for r in self.requests.values() if r.target == player]

    def outgoing(self, player):
        return next((r for r in self.requests.values() if r.sender == player), None)

    def session(self, player):
        return self.sessions.get(self.links.get(player))

    def _wake(self, player, reason=""):
        if player not in self.ready and not self.done:
            self.ready.append(player)
        if reason:
            self.last_result[player] = reason

    def action_mask(self, player):
        local = self.agents[player]
        mask = local.action_mask() | dict.fromkeys(COMM_KINDS, False)
        if self.done:
            return dict.fromkeys(mask, False)
        busy = player in self.futures
        if busy:
            for kind in ("measure", "score", "walk", "wait"):
                mask[kind] = False
            # Only forced notebook management interrupts local work.
            if local.pending is None:
                mask["delete"] = mask["summarize"] = False
        linked = player in self.links
        if linked:
            mask["score"] = mask["walk"] = False
        if self.communication:
            received = self.incoming(player)
            mask["request"] = (
                not linked and not busy and self.outgoing(player) is None and len(self.agents) > 1
            )
            mask["accept"] = bool(received) and not linked
            mask["reject"] = bool(received)
            mask["ignore"] = any(not r.ignored for r in received)
            mask["disconnect"] = linked
        mask["continue"] = busy
        # A full notebook must be managed before any new local/control action.
        if local.pending is not None:
            for kind in mask:
                if kind not in ("delete", "summarize"):
                    mask[kind] = False
        return mask

    def target_mask(self, player, kind):
        if kind == "request":
            # Other players' communication busy status is deliberately private.
            return np.arange(len(self.agents)) != player
        received = self.incoming(player)
        return np.array(
            [
                any(r.sender == i and (kind != "ignore" or not r.ignored) for r in received)
                for i in range(len(self.agents))
            ]
        )

    def _deliver(self, player, record):
        local = self.agents[player]
        if local.pending is not None:
            self.deliveries[player].append(record)
        else:
            local._insert(record)
        if local.pending is not None:
            self._wake(player, "notebook_full")

    def _drain(self, player):
        local = self.agents[player]
        while local.pending is None and self.deliveries[player]:
            local._insert(self.deliveries[player].popleft())

    def _snapshot(self, player):
        local = self.agents[player]
        return Record(
            "snapshot",
            local.position.copy(),
            self.time,
            None if local.last_direction is None else local.last_direction.copy(),
            batch=local.last_batch,
            source=player,
            source_m=local.config.measurement_time,
            source_l=local.config.latency,
        )

    def _start_sessions(self):
        for session in list(self.sessions.values()):
            if session.start is not None or any(
                i in self.futures and self.futures[i][1]["action"] == "walk"
                for i in (session.a, session.b)
            ):
                continue
            session.start = self.time
            session.end = self.time + max(
                self.agents[i].config.latency for i in (session.a, session.b)
            )
            session.snapshots = (self._snapshot(session.a), self._snapshot(session.b))
            self.stats["sessions_started"] += 1
            self.trace.append(
                {
                    "time": self.time,
                    "event": "session_start",
                    "a": session.a,
                    "b": session.b,
                    "end": session.end,
                }
            )
            for i in (session.a, session.b):
                if i in self.futures and self.futures[i][1]["action"] == "measure":
                    self.futures[i][1]["overlap"] = True
                if i in self.futures and self.futures[i][1]["action"] == "score":
                    del self.futures[i]
                    self.stats["score_cancelled"] += 1
                self._wake(i, "session_start")

    def _finish_session(self, sid, deliver):
        session = self.sessions.pop(sid)
        for i in (session.a, session.b):
            del self.links[i]
        if deliver:
            a, b = session.snapshots
            self._deliver(session.a, deepcopy(b))
            self._deliver(session.b, deepcopy(a))
            self.stats["sessions_completed"] += 1
            self.stats["snapshots"] += 2
        else:
            self.stats["disconnected"] += 1
        self.trace.append(
            {
                "time": self.time,
                "event": "session_end" if deliver else "disconnect",
                "a": session.a,
                "b": session.b,
            }
        )
        for i in (session.a, session.b):
            self._wake(i, "session_end" if deliver else "disconnect")

    def _respond(self, player, action):
        key = (action.target, player)
        request = self.requests[key]
        self.trace.append(
            {"time": self.time, "event": action.kind, "a": player, "b": action.target}
        )
        if action.kind == "ignore":
            request.ignored = True
            return
        del self.requests[key]
        sender = request.sender
        if action.kind == "reject":
            self.stats["rejected"] += 1
            self._wake(sender, "rejected")
            return
        sid = self.next_session
        self.next_session += 1
        self.sessions[sid] = Session(sender, player)
        self.links[sender] = self.links[player] = sid
        self.stats["accepted"] += 1
        # Reserve both endpoints immediately, including while a walk finishes.
        for other_key, other in list(self.requests.items()):
            if other.sender in (sender, player) or other.target in (sender, player):
                del self.requests[other_key]
                self.stats["403"] += 1
                self._wake(other.sender, "403")
        self._wake(sender, "accepted")
        self._start_sessions()

    def _complete_local(self, player, candidate, info):
        local = self.agents[player]
        if not info["completed"]:
            return
        kind = info["action"]
        # Commit deltas, preserving snapshots delivered during this local action.
        if kind == "walk":
            local.position, local.current_map = candidate.position.copy(), candidate.current_map
            local.rng = candidate.rng
        elif kind in ("measure", "score"):
            record = candidate.pending if candidate.pending is not None else candidate.records[-1]
            if kind == "measure":
                local.last_measurement = candidate.last_measurement
                local.last_direction = candidate.last_direction.copy()
                local.last_batch = candidate.last_batch
                local.rng = candidate.rng
                if info.get("overlap", False):
                    self.stats["overlap_measurements"] += 1
            self._deliver(player, record)

    def next_agent(self):
        if self.done:
            return None
        if not self.ready:
            times = [self.config.duration]
            times += [candidate.time for candidate, _ in self.futures.values()]
            times += [r.expires for r in self.requests.values()]
            times += [s.end for s in self.sessions.values() if s.start is not None]
            event_time = min(times)
            before = self.potential
            self.time = event_time
            # Publish ticks before this timestamp using previously finished measurements.
            for agent in self.agents:
                while agent.next_dashboard < event_time:
                    agent._publish()
                agent.time = event_time
            completed = [
                i for i, (candidate, _) in self.futures.items() if candidate.time == event_time
            ]
            for i in completed:
                candidate, info = self.futures.pop(i)
                self._complete_local(i, candidate, info)
                self.trace.append({"time": event_time, "agent": i, **info})
                self._wake(i)
            self._start_sessions()
            for sid, session in list(self.sessions.items()):
                if session.start is not None and session.end == event_time:
                    self._finish_session(sid, True)
            for key, request in list(self.requests.items()):
                if request.expires == event_time:
                    del self.requests[key]
                    self.stats["expired"] += 1
                    self.trace.append(
                        {
                            "time": self.time,
                            "event": "expired",
                            "a": request.sender,
                            "b": request.target,
                        }
                    )
                    self._wake(request.sender, "expired")
                    self._wake(request.target, "expired")
            for agent in self.agents:
                if agent.next_dashboard == event_time:
                    agent._publish()
            self.trace.append({"time": event_time, "team_reward": self.potential - before})
            if event_time >= self.config.duration:
                self.done = True
                self.ready.clear()
                self.futures.clear()
                self.requests.clear()
                self.sessions.clear()
                self.links.clear()
                for agent in self.agents:
                    agent.done = True
            # If an expiry only woke a busy player this still yields a valid continue decision.
        return None if self.done else self.ready[0]

    def submit(self, player, action):
        if self.done or not self.ready or self.ready[0] != player:
            raise ValueError("player does not have the next decision")
        if not self.action_mask(player).get(action.kind, False):
            raise ValueError(f"illegal team action: {action.kind}")
        if action.kind in ("request", "accept", "reject", "ignore") and (
            not isinstance(action.target, int)
            or not 0 <= action.target < len(self.agents)
            or not self.target_mask(player, action.kind)[action.target]
        ):
            raise ValueError("invalid communication target")
        if action.kind not in COMM_KINDS:
            candidate = deepcopy(self.agents[player])
            _, _, _, info = candidate.step(action)
        self.ready.popleft()
        self.last_result[player] = ""
        if action.kind not in ("accept", "reject", "ignore"):
            for request in self.incoming(player):
                request.ignored = True
        if action.kind == "request":
            self.stats["requests"] += 1
            self.trace.append(
                {"time": self.time, "event": "request", "a": player, "b": action.target}
            )
            if action.target in self.links:
                self.stats["403"] += 1
                self.trace.append(
                    {"time": self.time, "event": "403", "a": player, "b": action.target}
                )
                self.last_result[player] = "403"
            else:
                self.requests[(player, action.target)] = Request(
                    player, action.target, self.time + self.request_timeout
                )
                self._wake(action.target, "request")
            self._wake(player)
        elif action.kind in ("accept", "reject", "ignore"):
            self._respond(player, action)
            if player not in self.futures or any(not r.ignored for r in self.incoming(player)):
                self._wake(player)
        elif action.kind == "disconnect":
            self._finish_session(self.links[player], False)
        elif action.kind == "continue":
            pass
        elif info["dt"] == 0:
            self.agents[player] = candidate
            self._drain(player)
            self._wake(player)
        else:
            self.futures[player] = (
                candidate,
                {
                    "action": action.kind,
                    "overlap": self.session(player) is not None
                    and self.session(player).start is not None,
                    **info,
                },
            )
