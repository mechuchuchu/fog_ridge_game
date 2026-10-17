import numpy as np
import pytest
import torch

from fog_ridge import Action, Config, SoloEnv
from fog_ridge.policy import ActorCritic, decode_action, pack
from fog_ridge.ppo import collect, gae, load_checkpoint, save_checkpoint, update

torch.set_num_threads(1)


def test_semi_markov_gae_and_terminal_bootstrap():
    adv, returns = gae([1, 2], [0.3, 0.4], [2, 0], gamma=0.5, lam=1)
    np.testing.assert_allclose(returns, [1.5, 2], atol=1e-7)
    np.testing.assert_allclose(adv, [1.2, 1.6], atol=1e-7)
    _, undiscounted = gae([1, 2, -1], [0, 0, 0], [0, 3, 0.1], gamma=1, lam=1)
    np.testing.assert_allclose(undiscounted, [2, 1, -1])


def test_log_probability_replay_and_inactive_heads():
    torch.manual_seed(4)
    env = SoloEnv()
    env.step(Action("measure"))
    model = ActorCritic(width=16)
    data = pack(env.observe(), 20)
    for _ in range(30):
        action, old_logp, entropy = model.act(data)
        _, replay, _ = model.act(data, action)
        torch.testing.assert_close(old_logp, replay)
        assert torch.isfinite(entropy).all()
    # Only the kind likelihood is active for a score action.
    target = action.clone()
    target[0, 0] = 1
    _, first, _ = model.act(data, target)
    target[0, 1] = 7
    target[0, 6:] = 20
    _, second, _ = model.act(data, target)
    torch.testing.assert_close(first, second)


def test_set_encoder_permutation_equivariance():
    env = SoloEnv()
    for _ in range(3):
        env.step(Action("measure"))
        env.step(Action("walk", displacement=[0.2, -0.1]))
    model = ActorCritic(width=16)
    original = pack(env.observe(), 20)
    target = torch.tensor([[2, 0, 0, 2, 0, 0, -2, -5]], dtype=torch.float32)
    _, before, _ = model.act(original, target)
    env.records[0], env.records[2] = env.records[2], env.records[0]
    permuted = pack(env.observe(), 20)
    target[0, 2:4] = torch.tensor([2, 0])
    _, after, _ = model.act(permuted, target)
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)


def test_masked_actions_are_executable_including_forced_overflow():
    torch.manual_seed(5)
    env = SoloEnv(Config(duration=15, notebook_size=2))
    model = ActorCritic(capacity=2, width=16)
    for _ in range(1000):
        sample, logp, _ = model.act(pack(env.observe(), 2))
        assert torch.isfinite(logp).all()
        env.step(decode_action(env, sample[0]))
        if env.done:
            break
    assert env.done


def test_critic_parameters_cannot_affect_actor():
    model = ActorCritic(width=16)
    data = pack(SoloEnv().observe(), 20)
    before = model.act(data, deterministic=True)[0]
    with torch.no_grad():
        for parameter in model.critic.parameters():
            parameter.add_(100)
    after = model.act(data, deterministic=True)[0]
    torch.testing.assert_close(before, after)
    assert set(data) == {"tokens", "valid", "pair", "context", "legal"}


def test_ppo_updates_parameters_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(8)
    config = Config(duration=5, dimension=8, notebook_size=3)
    model = ActorCritic(capacity=3, width=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch, _ = collect(model, config, [2, 3], "cpu")
    before_actor = model.kind.weight.detach().clone()
    before_critic = model.critic[0].weight.detach().clone()
    metrics = update(model, optimizer, batch, epochs=2, minibatch=8)
    assert all(np.isfinite(value) for value in metrics.values())
    assert not torch.equal(before_actor, model.kind.weight)
    assert not torch.equal(before_critic, model.critic[0].weight)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, config, 2, 4, training={"gamma": 0.99})
    restored, restored_config, checkpoint = load_checkpoint(path)
    assert restored_config == config
    assert checkpoint["iteration"] == 2 and checkpoint["next_seed"] == 4
    assert checkpoint["training"] == {"gamma": 0.99}
    sample = pack(SoloEnv(config).observe(), 3)
    torch.testing.assert_close(
        model.act(sample, deterministic=True)[0], restored.act(sample, deterministic=True)[0]
    )
    # Restoring optimizer + RNG reproduces the next on-policy batch and update.
    resumed_optimizer = torch.optim.Adam(restored.parameters())
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    rng = checkpoint["torch_rng"]
    torch.set_rng_state(rng)
    next_batch, _ = collect(model, config, [4], "cpu")
    update(model, optimizer, next_batch, epochs=1, minibatch=8)
    torch.set_rng_state(rng)
    resumed_batch, _ = collect(restored, config, [4], "cpu")
    update(restored, resumed_optimizer, resumed_batch, epochs=1, minibatch=8)
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("dimension", [2, 16, 128])
def test_policy_generalizes_input_shape_across_dimensions(dimension):
    env = SoloEnv(Config(dimension=dimension))
    env.step(Action("measure"))
    model = ActorCritic(width=16)
    sample, _, _ = model.act(pack(env.observe(), 20))
    env.step(decode_action(env, sample[0]))
