"""Entropy-matched actor targets are scale-free and fail closed on invalid setups."""

import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from tests.unit.learning._algorithm_fixtures import BatchKind, DiscreteSacModel, _batch
from trackmaniarl.algorithms import StableDiscreteSoftActorCritic
from trackmaniarl.algorithms.sac_support import discrete_batch
from trackmaniarl.algorithms.sd_sac_objectives import entropy_matched_log_probabilities
from trackmaniarl.models.actors import CategoricalActor

MATCHED: dict[str, object] = {
    "actor_objective": "soft_q_forward_kl",
    "actor_temperature": "entropy_matched",
    "target_entropy": 0.8,
    "learn_entropy_coefficient": False,
    "backup_entropy_coefficient": 0.0,
    "entropy_penalty_coefficient": 0.0,
    "execution": {"device": "cpu", "torch_threads": 1},
}


class FixedValues(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.values.expand(len(observations), -1)


def _learner(model: nn.Module, **overrides: object) -> StableDiscreteSoftActorCritic:
    learner = StableDiscreteSoftActorCritic(model, **{**MATCHED, **overrides})
    learner.setup({"seed": 17})
    return learner


def _fixed_value_model(values: list[float]) -> nn.Module:
    model = nn.Module()
    model.actor = CategoricalActor(nn.Identity(), 4, len(values))
    model.q1, model.q2 = FixedValues(values), FixedValues(values)
    return model


def _entropy(logs: torch.Tensor) -> torch.Tensor:
    return -(logs.exp() * logs).sum(1)


@pytest.mark.parametrize("scale", [1e-9, 1e-4, 1e-2, 1.0, 100.0, 1e6])
def test_target_has_requested_entropy_at_any_value_scale(scale: float) -> None:
    generator = torch.Generator().manual_seed(17)
    values = torch.randn(64, 78, generator=generator, dtype=torch.float64) * scale
    logs, alpha = entropy_matched_log_probabilities(values, 0.8)
    assert bool(torch.isfinite(logs).all() & torch.isfinite(alpha).all() & (alpha > 0).all())
    torch.testing.assert_close(_entropy(logs), torch.full((64,), 0.8, dtype=torch.float64))


@pytest.mark.parametrize(("scale", "shift"), [(1e-3, -5.0), (7.0, 100.0), (1e3, 0.0)])
def test_target_is_invariant_to_positive_affine_value_changes(scale: float, shift: float) -> None:
    generator = torch.Generator().manual_seed(29)
    values = torch.randn(32, 78, generator=generator, dtype=torch.float64)
    reference, _ = entropy_matched_log_probabilities(values, 0.8)
    changed, _ = entropy_matched_log_probabilities(values * scale + shift, 0.8)
    torch.testing.assert_close(changed.exp(), reference.exp(), rtol=0, atol=1e-9)


def test_flat_rows_are_uniform_and_tied_maxima_share_the_mass() -> None:
    flat, _ = entropy_matched_log_probabilities(torch.full((2, 78), 3.25), 0.8)
    torch.testing.assert_close(flat.exp(), torch.full((2, 78), 1 / 78, dtype=torch.float64))
    tied = torch.full((1, 78), -1.0)
    tied[0, :3] = 0.0
    logs, _ = entropy_matched_log_probabilities(tied, 0.8)
    # Three equal maxima cannot carry less than log(3) nats, whatever the target.
    assert float(_entropy(logs)[0]) == pytest.approx(math.log(3), abs=1e-6)
    assert bool(torch.isfinite(logs).all())


@pytest.mark.parametrize("target", [0.0, -0.1, math.log(78), 5.0])
def test_unreachable_target_entropy_rejected(target: float) -> None:
    with pytest.raises(ValueError, match="target_entropy"):
        entropy_matched_log_probabilities(torch.zeros(1, 78), target)


def test_nonfinite_values_rejected() -> None:
    values = torch.zeros(1, 78)
    values[0, 5] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        entropy_matched_log_probabilities(values, 0.8)


def test_actor_target_ignores_the_scalar_alpha() -> None:
    values = [0.0] * 78
    values[35] = 0.0021
    model = _fixed_value_model(values)
    learner = _learner(model)
    batch = replace(_batch(BatchKind.DISCRETE), observations=torch.zeros(8, 4))
    prepared = discrete_batch(learner._batch(batch))
    expected, _ = entropy_matched_log_probabilities(torch.tensor([values] * 8), 0.8)
    logs = model.actor.log_probabilities(prepared.observations).detach()
    expected_loss = float(-(expected.exp() * logs).sum(1).mean())
    for alpha in (0.2, 0.01):
        step = learner._actor_step(prepared, torch.tensor(alpha))
        assert float(step.loss.detach()) == pytest.approx(expected_loss, rel=1e-6)
        assert float(step.diagnostics["policy/soft_q_entropy"]) == pytest.approx(0.8, abs=1e-6)
        assert float(step.diagnostics["policy/matched_alpha_mean"]) > 0


def test_start_like_action_gap_sharpens_only_with_entropy_matched_target() -> None:
    """CP33856 start: mean-Q top-2 gap 0.0021 under alpha 0.01 leaves the target flat."""
    values = [0.0] * 78
    values[35] = 0.0021
    batch = replace(_batch(BatchKind.DISCRETE), observations=torch.zeros(8, 4))
    final = {}
    for temperature in ("learned", "entropy_matched"):
        torch.manual_seed(17)
        model = _fixed_value_model(values)
        overrides: dict[str, object] = {"actor_learning_rate": 0.05}
        if temperature == "learned":
            overrides.update(actor_temperature="learned", backup_entropy_coefficient=None)
        learner = _learner(model, **overrides)
        prepared = discrete_batch(learner._batch(batch))
        for _ in range(300):
            step = learner._actor_step(prepared, torch.tensor(0.01))
            learner._optimize(step.loss, learner.actor_optimizer)
        final[temperature] = model.actor.log_probabilities(prepared.observations).detach()
    assert float(_entropy(final["learned"]).mean()) > 4.2
    assert float(_entropy(final["entropy_matched"]).mean()) < 1.5
    assert bool((final["entropy_matched"].argmax(1) == 35).all())


@pytest.mark.parametrize("coefficient", [None, 0.0, 0.05])
def test_critic_backup_uses_configured_entropy_coefficient(coefficient: float | None) -> None:
    learner = StableDiscreteSoftActorCritic(
        DiscreteSacModel(),
        backup_entropy_coefficient=coefficient,
        entropy_penalty_coefficient=0,
        execution={"device": "cpu", "torch_threads": 1},
    )
    learner.setup({"seed": 17})
    batch = discrete_batch(learner._batch(_batch(BatchKind.DISCRETE)))
    used = 0.2 if coefficient is None else coefficient
    with torch.no_grad():
        logs = learner.model.actor.log_probabilities(batch.next_observations).float()
        target = learner.target_model
        q = 0.5 * target.q1(batch.next_observations) + 0.5 * target.q2(batch.next_observations)
        continuation = (logs.exp() * (q - used * logs)).sum(1)
    expected = batch.rewards + batch.discounts * continuation
    torch.testing.assert_close(learner._critic_targets(batch, torch.tensor(0.2)), expected)


def test_entropy_matched_update_is_finite_without_alpha_optimizer() -> None:
    learner = _learner(DiscreteSacModel())
    metrics, priorities = learner.update(_batch(BatchKind.DISCRETE))
    assert learner.log_alpha is None
    assert learner.alpha_optimizer is None
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["policy/soft_q_entropy"] == pytest.approx(0.8, abs=1e-6)
    assert metrics["policy/matched_alpha_mean"] > 0
    assert priorities.transition_ids == list(range(8))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"actor_objective": "sac"}, "soft_q_forward_kl"),
        ({"target_entropy": None}, "target_entropy"),
        ({"learn_entropy_coefficient": True}, "learn_entropy_coefficient"),
        ({"backup_entropy_coefficient": None}, "backup_entropy_coefficient"),
        ({"backup_entropy_coefficient": -0.1}, "backup_entropy_coefficient"),
        ({"backup_entropy_coefficient": float("nan")}, "backup_entropy_coefficient"),
        ({"actor_temperature": "fixed"}, "actor_temperature"),
    ],
)
def test_invalid_entropy_matched_configuration_rejected(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        StableDiscreteSoftActorCritic(DiscreteSacModel(), **{**MATCHED, **overrides})


def test_default_options_keep_the_canonical_checkpoint_contract() -> None:
    learner = StableDiscreteSoftActorCritic(
        DiscreteSacModel(), execution={"device": "cpu", "torch_threads": 1}
    )
    learner.setup({"seed": 17})
    options = learner.state_dict()["sd_sac_options"]
    assert "actor_temperature" not in options
    assert "backup_entropy_coefficient" not in options


def test_checkpoint_pins_entropy_matched_and_backup_options() -> None:
    state = _learner(DiscreteSacModel()).state_dict()
    _learner(DiscreteSacModel()).load_state_dict(state)
    learned = _learner(
        DiscreteSacModel(), actor_temperature="learned", backup_entropy_coefficient=None
    )
    with pytest.raises(ValueError, match="options do not match"):
        learned.load_state_dict(state)
    backup_only = StableDiscreteSoftActorCritic(
        DiscreteSacModel(),
        backup_entropy_coefficient=0.0,
        entropy_penalty_coefficient=0,
        execution={"device": "cpu", "torch_threads": 1},
    )
    backup_only.setup({"seed": 17})
    legacy = dict(backup_only.state_dict())
    del legacy["sd_sac_options"]
    with pytest.raises(ValueError, match="legacy checkpoint"):
        backup_only.load_state_dict(legacy)
