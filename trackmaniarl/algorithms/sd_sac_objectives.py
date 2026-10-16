"""Discrete policy numerics and detached diagnostics; no optimizer or environment."""

from __future__ import annotations

import math
from typing import Any

import torch


def categorical_statistics(actor: Any, observations: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefer log-softmax; retain the probabilities-only custom actor contract."""
    if callable(getattr(actor, "log_probabilities", None)):
        log_probabilities = actor.log_probabilities(observations).float()
        return log_probabilities.exp(), log_probabilities
    probabilities = actor.probabilities(observations).float()
    return probabilities, probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()


def soft_q_log_probabilities(q_values: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    if q_values.ndim != 2 or not all(q_values.shape) or alpha.numel() != 1:
        raise ValueError("soft-Q requires nonempty [batch, actions] values and scalar alpha")
    values, temperature = q_values.float(), alpha.float().reshape(())
    if not bool(torch.isfinite(values).all() & torch.isfinite(temperature) & (temperature > 0)):
        raise ValueError("soft-Q requires finite float32 values and finite positive float32 alpha")
    # Every finite float32 Q difference / positive float32 alpha fits in float64.
    # Keep log probabilities in float64: casting them back can produce -inf and
    # NaN in 0 * log(p) diagnostics, even when softmax probabilities are valid.
    values = values.double()
    advantages = values - values.max(1, keepdim=True).values
    return (advantages / temperature.double()).log_softmax(dim=1)


def entropy_matched_log_probabilities(
    q_values: torch.Tensor, target_entropy: float, iterations: int = 40
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft-Q targets whose per-state temperature gives exactly ``target_entropy``.

    A scalar alpha is measured in reward units: action gaps far below it leave the
    target near uniform, gaps far above it make the target nearly greedy. Solving
    each row's temperature makes the target invariant to ``Q -> c * Q + b`` for
    ``c > 0``. Returns float64 log probabilities and the ``[batch, 1]`` temperatures.
    """
    if q_values.ndim != 2 or not all(q_values.shape):
        raise ValueError("entropy-matched soft-Q requires nonempty [batch, actions] values")
    actions = q_values.shape[1]
    if not 0.0 < target_entropy < math.log(actions):
        raise ValueError("entropy-matched target_entropy must lie in (0, log(action_count))")
    values = q_values.detach().double()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("entropy-matched soft-Q requires finite values")
    advantages = values - values.max(1, keepdim=True).values
    spread = -advantages.min(1, keepdim=True).values
    flat = spread <= 0
    scale = torch.where(flat, torch.ones_like(spread), spread)
    # Entropy grows monotonically with temperature: bisect log(alpha / spread)
    # between a nearly greedy and a nearly uniform target, independently per row.
    low = torch.full_like(spread, -12.0)
    high = torch.full_like(spread, 12.0)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        logs = (advantages / (scale * middle.exp())).log_softmax(dim=1)
        too_sharp = -(logs.exp() * logs).sum(1, keepdim=True) < target_entropy
        low = torch.where(too_sharp, middle, low)
        high = torch.where(too_sharp, high, middle)
    alpha = scale * (0.5 * (low + high)).exp()
    logs = (advantages / alpha).log_softmax(dim=1)
    uniform = torch.full_like(logs, -math.log(actions))
    return torch.where(flat, uniform, logs), alpha


def terminal_value_loss(  # noqa: PLR0913 -- Explicit target, terminal mask and importance weights.
    critics: tuple[torch.Tensor, torch.Tensor],
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Terminal-only regression to known rewards, without bootstrap or clipping.

    Normalize over terminal samples so their influence does not vanish in a long
    episode batch. Truncations are excluded: their return is unknown.
    """
    mask = terminated.reshape(-1).double()
    if weights is not None:
        mask = mask * weights.reshape(-1).double()
    mass = mask.sum()
    # Normalize before multiplying errors: a denominator floor changes the
    # objective for subnormal weights, and dividing the loss by that tiny mass
    # can overflow its backward reciprocal. An empty terminal mass stays zero.
    normalized = mask / torch.where(mass > 0, mass, torch.ones_like(mass))
    q1, q2 = critics
    errors = (q1 - rewards).square() + (q2 - rewards).square()
    return (normalized.to(errors.dtype) * errors).sum()


@torch.no_grad()
def actor_diagnostics(
    policy: tuple[torch.Tensor, torch.Tensor],
    critics: tuple[torch.Tensor, torch.Tensor],
    target_log_probabilities: torch.Tensor,
) -> dict[str, torch.Tensor]:
    probabilities, log_probabilities = policy
    q1, q2 = critics
    q_values = 0.5 * q1 + 0.5 * q2
    actor_actions = probabilities.argmax(1)
    best_values = q_values.max(1).values
    entropy = -(probabilities * log_probabilities).sum(1)
    target_probabilities = target_log_probabilities.exp()
    top_probabilities = probabilities.topk(2, dim=1).values
    top_values = q_values.topk(2, dim=1).values
    critic_actions = q_values.argmax(1)
    return {
        "policy/max_probability": probabilities.max(1).values.mean(),
        # Entropy alone can hide a broad policy whose greedy maximum is wrong.
        "policy/greedy_probability_margin": (
            top_probabilities[:, 0] - top_probabilities[:, 1]
        ).mean(),
        "policy/critic_greedy_probability": probabilities.gather(1, critic_actions[:, None]).mean(),
        "policy/soft_q_cross_entropy": -(target_probabilities * log_probabilities).sum(1).mean(),
        "policy/q_greedy_agreement": (actor_actions == q_values.argmax(1)).float().mean(),
        "policy/q_greedy_regret": (
            best_values - q_values.gather(1, actor_actions[:, None]).squeeze(1)
        ).mean(),
        "policy/q_expected_regret": (best_values - (probabilities * q_values).sum(1)).mean(),
        "policy/soft_q_reverse_kl": (probabilities * (log_probabilities - target_log_probabilities))
        .sum(1)
        .mean(),
        "policy/soft_q_forward_kl": (
            target_probabilities * (target_log_probabilities - log_probabilities)
        )
        .sum(1)
        .mean(),
        "policy/soft_q_entropy": -(target_probabilities * target_log_probabilities).sum(1).mean(),
        "policy/entropy_min": entropy.min(),
        "critic/action_value_spread": (best_values - q_values.min(1).values).mean(),
        "critic/greedy_value_margin": (top_values[:, 0] - top_values[:, 1]).mean(),
        "critic/q_greedy_agreement": (q1.argmax(1) == q2.argmax(1)).float().mean(),
        "critic/all_action_disagreement": (q1 - q2).abs().mean(),
    }
