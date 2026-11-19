"""
Tests for the GPT feedforward network variants (ReLU^2 and SwiGLU).

Run: python -m pytest tests/test_gpt.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig, MLP, SwiGLU


def tiny_config(**overrides):
    """A small but complete GPT config that runs on CPU."""
    kwargs = dict(
        sequence_len=32, vocab_size=64, n_layer=1, n_head=2, n_kv_head=2,
        n_embd=96, window_pattern="L",
    )
    kwargs.update(overrides)
    return GPTConfig(**kwargs)


@pytest.mark.parametrize("mlp_type", ["relu2", "swiglu"])
def test_mlp_parameter_count_matches_4x(mlp_type):
    """SwiGLU uses 8/3x expansion so its parameter count matches the 4x ReLU^2 MLP."""
    config = GPTConfig(n_embd=96)
    mlp = MLP(config) if mlp_type == "relu2" else SwiGLU(config)
    assert sum(p.numel() for p in mlp.parameters()) == 8 * config.n_embd**2


def test_relu2_forward_matches_reference():
    config = GPTConfig(n_embd=96)
    mlp = MLP(config)
    torch.manual_seed(0)
    for p in mlp.parameters():
        torch.nn.init.normal_(p)
    x = torch.randn(2, 5, config.n_embd)
    expected = mlp.c_proj(F.relu(mlp.c_fc(x)).square())
    assert torch.allclose(mlp(x), expected, atol=1e-5, rtol=1e-5)


def test_swiglu_forward_matches_reference():
    config = GPTConfig(n_embd=96)
    mlp = SwiGLU(config)
    torch.manual_seed(0)
    for p in mlp.parameters():
        torch.nn.init.normal_(p)
    x = torch.randn(2, 5, config.n_embd)
    expected = mlp.c_proj(F.silu(mlp.c_gate(x)) * mlp.c_up(x))
    assert torch.allclose(mlp(x), expected, atol=1e-5, rtol=1e-5)


def test_default_model_uses_relu2():
    model = GPT(tiny_config())
    assert isinstance(model.transformer.h[0].mlp, MLP)


def test_swiglu_model_and_parameter_parity():
    relu2_model = GPT(tiny_config())
    swiglu_model = GPT(tiny_config(mlp_type="swiglu"))
    assert isinstance(swiglu_model.transformer.h[0].mlp, SwiGLU)
    # Attention is unchanged, so equal transformer params means the MLPs match too.
    assert swiglu_model.num_scaling_params()["transformer_matrices"] == \
        relu2_model.num_scaling_params()["transformer_matrices"]
    assert swiglu_model.estimate_flops() == relu2_model.estimate_flops()


def test_invalid_mlp_type_is_rejected():
    with pytest.raises(AssertionError, match="mlp_type"):
        GPT(tiny_config(mlp_type="geglu"))


def test_old_checkpoint_config_defaults_to_relu2():
    from nanochat.checkpoint_manager import _patch_missing_config_keys
    config = {}
    _patch_missing_config_keys(config)
    assert config["window_pattern"] == "L"
    assert config["mlp_type"] == "relu2"


@pytest.mark.parametrize("mlp_type", ["relu2", "swiglu"])
def test_model_forward_backward(mlp_type):
    config = tiny_config(mlp_type=mlp_type)
    model = GPT(config)
    model.init_weights()
    torch.manual_seed(0)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    loss = model(idx, targets)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
