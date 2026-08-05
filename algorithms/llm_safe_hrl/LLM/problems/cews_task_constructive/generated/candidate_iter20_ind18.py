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
    """
    v2 evolved priority: Deadline-hardened, energy-aware, and starvation-resistant.
    Combines Parent 2's global normalization and smooth slack gating with Parent 1's
    slack-constrained criticality density (SCD) and violation-offset safety guard.
    Key improvements:
      - Adds SCD = upward_rank * sigmoid(slack/tau) / (exec+comm) to prioritize critical work only when slack permits
      - Retains Parent 2's robust global MAD-based normalization for hierarchy preservation
      - Uses tanh-based urgency (Parent 2) but adds hard violation penalty (Parent 1)
      - Simplified fairness: sqrt(wait) * exp(-uncertainty) gated by positive slack (Parent 2), capped and normalized
      - All operations zero/Nan/inf protected; deterministic; returns shape-(N,)
    """
    eps = 1e-08
    N = len(np.atleast_1d(slack))
    if N == 0:
        return np.array([], dtype=float)

    # Safe input conversion and nan/inf handling
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=eps, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1000000.0, neginf=-1000000.0)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=eps, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    base_signals = []

    # 1. Urgency: tanh(-slack/tau) — sharp near deadline, bounded, smooth
    tau_urg = 0.5
    urgency_raw = np.tanh(-np.clip(slack, -100.0, 100.0) / tau_urg)
    base_signals.append(urgency_raw)

    # 2. Slack-Constrained Criticality Density (SCD): upward_rank * gate / (exec+comm)
    # Prioritizes critical path only when slack allows; avoids over-scheduling non-urgent critical work
    slack_gate_scd = 1.0 / (1.0 + np.exp(-slack / (tau_urg + eps)))
    exec_comm_sum = min_exec_time + min_comm_time + eps
    scd_raw = upward_rank * slack_gate_scd / exec_comm_sum
    base_signals.append(scd_raw)

    # 3. Energy efficiency: 1/(energy+eps), gated by slack to favor efficiency when safe
    energy_efficiency = 1.0 / (min_incremental_energy + eps)
    slack_gate_energy = 1.0 / (1.0 + np.exp(-slack / 1.0))  # sigmoid(slack)
    seer_gated = energy_efficiency * slack_gate_energy
    base_signals.append(seer_gated)

    # 4. Fairness: sqrt(wait) * exp(-uncertainty), only when slack > 0, capped
    wait_safe = np.maximum(ready_wait_time, 0.0)
    sqrt_wait = np.sqrt(wait_safe + eps)
    fairness_raw = np.where(slack > 0.0, sqrt_wait * np.exp(-uncertainty), 0.0)
    fairness_clipped = np.clip(fairness_raw, 0.0, 0.25)
    base_signals.append(fairness_clipped)

    # Global normalization context: use MAD over all base signals
    if N == 1:
        norm_scale = eps
        norm_shift = 0.0
    else:
        all_base = np.concatenate([s.reshape(-1) for s in base_signals])
        med = np.median(all_base)
        abs_dev = np.abs(all_base - med)
        mad = np.median(abs_dev)
        norm_scale = mad if mad > eps else (np.max(all_base) - np.min(all_base) + eps)
        norm_shift = med

    def global_normalize(x):
        return (x - norm_shift) / (norm_scale + eps)

    norm_urgency = global_normalize(urgency_raw)
    norm_scd = global_normalize(scd_raw)
    norm_seer = global_normalize(seer_gated)
    norm_fairness = global_normalize(fairness_clipped)

    # Hierarchical weights: urgency dominates, then SCD (critical path under slack), then energy, then fairness
    urgency_term = -5.0 * norm_urgency
    scd_term = -3.0 * norm_scd
    seer_term = -1.6 * norm_seer
    fairness_term = -0.6 * norm_fairness

    # Hard violation penalty: assign massively negative score to violate tasks to enforce DDL hardness
    violation_offset = np.where(slack < 0, -1000000000.0, 0.0)

    score = urgency_term + scd_term + seer_term + fairness_term + violation_offset

    # Final sanitization: finite, clipped, shape-(N,)
    score = np.nan_to_num(score, nan=1000000000.0, posinf=1000000000.0, neginf=-1000000000.0)
    score = np.clip(score, -1000000000.0, 1000000000.0)
    return score.reshape(-1)
