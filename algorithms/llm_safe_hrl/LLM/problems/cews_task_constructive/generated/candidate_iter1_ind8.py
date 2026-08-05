import numpy as np


def get_task_priority_v2(
    min_exec_time,
    min_comm_time,
    min_incremental_energy,
    slack,
    upward_rank,
    remaining_work,
    ready_wait_time,
    uncertainty
):
    """Novel priority rule emphasizing deadline-criticality, risk-aware energy efficiency,
    and starvation prevention via adaptive normalization and slack-driven gating.

    Key innovations:
    - Uses *slack-gated* normalization: features are scaled only over tasks with slack < 0
      (imminent risk) or globally if all slack >= 0 — preventing distortion from outliers.
    - Introduces 'criticality-energy ratio': upward_rank / (min_incremental_energy + eps)
      to favor high-impact-low-energy tasks *only when feasible*, avoiding division by zero.
    - Replaces linear urgency with a smooth, bounded sigmoid-like slack penalty:
      exp(-slack / (|mean_slack| + eps)) for slack >= 0, and linear penalty for slack < 0.
    - Integrates waiting time as a *relative* anti-starvation term: normalized by max wait,
      capped to avoid dominating critical paths.
    - Combines execution+communication as 'total_latency' and normalizes it *jointly*
      with energy and criticality — preserving correlation while ensuring scale parity.
    - All operations use epsilon-safe, finite-preserving primitives; no logs, no unbounded exp.
    """
    eps = 1e-8

    # Ensure float64 & copy inputs (no in-place mutation)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    # === 1. Slack-based gating for robust normalization ===
    # Use only negative-slack tasks for normalization if any exist; else use all
    has_risk = np.any(slack < 0)
    norm_mask = slack < 0 if has_risk else np.ones(N, dtype=bool)
    
    # Compute robust denominators: mean absolute value over norm_mask
    def safe_mean_abs(x):
        masked = x[norm_mask]
        return np.mean(np.abs(masked)) + eps if len(masked) > 0 else eps
    
    exec_norm = safe_mean_abs(min_exec_time)
    comm_norm = safe_mean_abs(min_comm_time)
    energy_norm = safe_mean_abs(min_incremental_energy)
    rank_norm = safe_mean_abs(upward_rank)
    work_norm = safe_mean_abs(remaining_work)
    wait_norm = np.maximum(np.max(ready_wait_time), eps)

    # === 2. Slack penalty: strong linear penalty for negative slack, soft decay for positive ===
    # For slack < 0: linear penalty proportional to |slack| → higher priority (lower score)
    # For slack >= 0: exponential decay: exp(-slack / (mean_slack_pos + eps)) → lower penalty
    mean_slack_pos = np.mean(slack[slack >= 0]) + eps if np.any(slack >= 0) else eps
    slack_penalty = np.where(
        slack < 0,
        -slack,  # raw lateness magnitude → penalize heavily
        np.exp(-slack / mean_slack_pos)  # decays from 1.0 to ~0 as slack grows
    )

    # === 3. Criticality-energy efficiency ratio: higher = better (per unit energy) ===
    # Only meaningful when energy > 0; clamp denominator to avoid instability
    ce_ratio = upward_rank / (min_incremental_energy + eps)
    # Normalize ce_ratio using same norm_mask logic
    ce_norm = safe_mean_abs(ce_ratio)
    ce_score = ce_ratio / ce_norm  # higher ce_ratio → lower priority score (we want *high* ce_ratio)

    # === 4. Total latency (exec + comm) normalized jointly ===
    total_latency = min_exec_time + min_comm_time
    latency_norm = safe_mean_abs(total_latency)
    latency_score = total_latency / latency_norm

    # === 5. Anti-starvation: relative wait time, capped at 0.5 to prevent dominance ===
    wait_score = np.clip(ready_wait_time / wait_norm, 0.0, 0.5)

    # === 6. Uncertainty-weighted energy: reward low-uncertainty low-energy tasks ===
    # Prefer low energy *and* low uncertainty → combine multiplicatively (both small → better)
    # Use uncertainty normalized over norm_mask
    unc_norm = safe_mean_abs(uncertainty)
    unc_score = uncertainty / unc_norm
    # Penalize high uncertainty *and* high energy → (energy_norm * unc_score) promotes safe-low-energy
    risk_adjusted_energy = (min_incremental_energy / energy_norm) * (1.0 + unc_score)

    # === 7. Composite score: smaller = higher priority ===
    # Negative weights for beneficial terms (high ce_ratio, low latency, low risk_energy, low wait)
    # Positive weight for slack_penalty: since slack_penalty is *larger* when urgent → we want to *reduce* score
    # So use -slack_penalty to push urgent tasks down in score
    score = (
        + 0.30 * latency_score                # penalize high latency
        + 0.35 * risk_adjusted_energy       # penalize risky+expensive
        - 0.40 * ce_score                   # reward criticality-per-energy
        - 0.25 * slack_penalty              # strongly reward urgency (negative slack → large -penalty)
        + 0.10 * wait_score                 # mild boost for long-waiting (anti-starvation)
        - 0.05 * (upward_rank / rank_norm)  # slight preference for higher rank (not dominant)
    )

    # Final safeguard: ensure finite output, no NaN/inf
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    return score
