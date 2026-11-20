"""
Tests for the GPT feedforward network variants (ReLU^2 and SwiGLU).

Run: python -m pytest tests/test_gpt.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig, MLP, SwiGLU, LowRankLinear, LOW_RANK_ACTIVATIONS


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
    assert config["ffn_low_rank_r"] == 0
    assert config["ffn_low_rank_variant"] == "direct"
    assert config["ffn_low_rank_activation"] == "silu"


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


# -----------------------------------------------------------------------------
# Low-rank FFN experiment


def low_rank_mlp(mlp_type, r, variant="direct", activation="silu", n_embd=96):
    config = GPTConfig(
        n_embd=n_embd, mlp_type=mlp_type, ffn_low_rank_r=r,
        ffn_low_rank_variant=variant, ffn_low_rank_activation=activation,
    )
    if mlp_type == "relu2":
        return MLP(config)
    return SwiGLU(config)


def expected_low_rank_params(in_features, out_features, r):
    d = min(in_features, out_features) // r
    return d * (in_features + out_features)


@pytest.mark.parametrize("r", [2, 4, 8])
@pytest.mark.parametrize("mlp_type", ["relu2", "swiglu"])
def test_low_rank_shapes(mlp_type, r):
    n_embd = 96
    mlp = low_rank_mlp(mlp_type, r)
    expected_rank = n_embd // r
    for module in mlp.modules():
        if isinstance(module, LowRankLinear):
            assert module.rank == expected_rank
            assert module.a.shape == (module.out_features, expected_rank)
            assert module.b.shape == (expected_rank, module.in_features)


@pytest.mark.parametrize("r", [2, 4, 8])
@pytest.mark.parametrize("mlp_type", ["relu2", "swiglu"])
def test_low_rank_parameter_count(mlp_type, r):
    """Every FFN projection is factored, so the counts follow d*(M+N)."""
    n_embd = 96
    mlp = low_rank_mlp(mlp_type, r)
    if mlp_type == "relu2":
        expected = 2 * expected_low_rank_params(n_embd, 4 * n_embd, r)
    else:
        hidden_dim = (8 * n_embd) // 3
        expected = (
            expected_low_rank_params(n_embd, hidden_dim, r)  # c_gate
            + expected_low_rank_params(n_embd, hidden_dim, r)  # c_up
            + expected_low_rank_params(hidden_dim, n_embd, r)  # c_proj
        )
    assert sum(p.numel() for p in mlp.parameters()) == expected


def test_low_rank_disabled_keeps_plain_linears():
    mlp = low_rank_mlp("relu2", r=0)
    assert isinstance(mlp.c_fc, torch.nn.Linear)
    assert not isinstance(mlp.c_fc, LowRankLinear)
    assert mlp.c_fc.weight.shape == (4 * 96, 96)


def test_low_rank_direct_matches_full_weight():
    layer = LowRankLinear(96, 384, 24, variant="direct", activation="silu")
    torch.manual_seed(0)
    with torch.no_grad():
        layer.a.normal_()
        layer.b.normal_()
    x = torch.randn(2, 5, 96)
    # A(Bx) == (AB)x, i.e. the factorization is an exact rank-d linear map.
    expected = F.linear(x, layer.a @ layer.b)
    # Different matmul association orders, so compare with a modest tolerance.
    assert torch.allclose(layer(x), expected, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("activation", sorted(LOW_RANK_ACTIVATIONS))
def test_low_rank_nonlinear_matches_reference(activation):
    layer = LowRankLinear(
        96, 384, 24, variant="nonlinear", activation=activation,
    )
    torch.manual_seed(0)
    with torch.no_grad():
        layer.a.normal_()
        layer.b.normal_()
    x = torch.randn(2, 5, 96)
    hidden = LOW_RANK_ACTIVATIONS[activation](F.linear(x, layer.b))
    expected = F.linear(hidden, layer.a)
    assert torch.allclose(layer(x), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mlp_type", ["relu2", "swiglu"])
@pytest.mark.parametrize("variant", ["direct", "nonlinear"])
def test_low_rank_output_starts_at_zero(mlp_type, variant):
    """The zero-initialized output projection is preserved: A starts at zero."""
    mlp = low_rank_mlp(mlp_type, r=4, variant=variant)
    mlp.init_weights(s=3**0.5 * 96**-0.5)
    torch.manual_seed(0)
    x = torch.randn(2, 5, 96)
    assert torch.count_nonzero(mlp(x)) == 0
    first = mlp.c_fc if mlp_type == "relu2" else mlp.c_gate
    assert torch.count_nonzero(first.b) > 0
    assert torch.count_nonzero(first.a) > 0


@pytest.mark.parametrize("mlp_type", ["relu2", "swiglu"])
def test_low_rank_model_params_and_flops(mlp_type):
    base = GPT(tiny_config(mlp_type=mlp_type))
    low = GPT(tiny_config(mlp_type=mlp_type, ffn_low_rank_r=2))
    base_mlp_params = sum(p.numel() for p in base.transformer.h[0].mlp.parameters())
    low_mlp_params = sum(p.numel() for p in low.transformer.h[0].mlp.parameters())
    assert low_mlp_params < base_mlp_params
    # Attention is unchanged; only the FFN matmul parameters (and thus FLOPs) shrink.
    assert low.num_matmul_params() == base.num_matmul_params() - base_mlp_params + low_mlp_params
    assert base.estimate_flops() - low.estimate_flops() == 6 * (base_mlp_params - low_mlp_params)


@pytest.mark.parametrize("r", [2, 8])
@pytest.mark.parametrize("variant", ["direct", "nonlinear"])
@pytest.mark.parametrize("mlp_type", ["relu2", "swiglu"])
def test_low_rank_model_forward_backward(mlp_type, variant, r):
    config = tiny_config(
        mlp_type=mlp_type, ffn_low_rank_r=r, ffn_low_rank_variant=variant,
    )
    model = GPT(config)
    model.init_weights()
    torch.manual_seed(0)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    loss = model(idx, targets)
    assert torch.isfinite(loss)
    loss.backward()
    factor_grads = [
        p.grad for p in model.parameters() if p.grad is not None and p.ndim == 2
    ]
    assert factor_grads
    assert all(torch.isfinite(g).all() for g in factor_grads)
    # The zero-initialized output projection still gets an A gradient from the loss
    # (its B gradient is zero on the first step, as with a zero full-rank weight).
    mlp = model.transformer.h[0].mlp
    assert mlp.c_proj.a.grad is not None and torch.count_nonzero(mlp.c_proj.a.grad) > 0


def test_low_rank_checkpoint_round_trip():
    config = tiny_config(ffn_low_rank_r=4, ffn_low_rank_variant="nonlinear")
    model = GPT(config)
    model.init_weights()
    state = model.state_dict()
    assert "transformer.h.0.mlp.c_fc.a" in state
    assert "transformer.h.0.mlp.c_fc.b" in state
    loaded = GPT(config)
    loaded.init_weights()
    loaded.load_state_dict(state, strict=True)
    for key, value in state.items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_invalid_low_rank_config_is_rejected():
    with pytest.raises(AssertionError, match="ffn_low_rank_r"):
        GPT(tiny_config(ffn_low_rank_r=-1))
    with pytest.raises(AssertionError, match="ffn_low_rank_variant"):
        GPT(tiny_config(ffn_low_rank_r=2, ffn_low_rank_variant="bilinear"))
    with pytest.raises(AssertionError, match="ffn_low_rank_activation"):
        GPT(tiny_config(ffn_low_rank_r=2, ffn_low_rank_activation="mish"))


def test_low_rank_requires_divisible_rank():
    with pytest.raises(AssertionError, match="must divide"):
        low_rank_mlp("relu2", r=7, n_embd=96)
