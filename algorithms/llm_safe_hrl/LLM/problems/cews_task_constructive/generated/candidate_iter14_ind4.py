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
    Hybrid priority rule v2: merges Parent 2's relative slack robustness and risk-adaptive energy scaling
    with Parent 1's starvation control stability and slack-gated criticality, while enhancing numerical safety,
    deadline fidelity, and energy-context sensitivity.

    Key innovations:
      - Unified urgency: combines *relative slack ratio* (Parent 2) with *slack-gap amplification* (Parent 1)
        via smooth sigmoid gating for monotonic, bounded urgency [0,1].
      - Criticality-energy synergy: upward_rank / (min_incremental_energy * (1+uncertainty)^alpha) with alpha
        now dynamically modulated by both slack *and* remaining_work to avoid over-penalizing large workflows.
      - Starvation guard: work-normalized wait pressure gated by *both* slack > 0 AND low latency, using
        adaptive quantile threshold (midpoint method) and explicit finite bounds.
      - Energy scaling: context-aware — only scales energy by remaining_work when upward_rank is high *and*
        slack is positive, preventing energy over-prioritization in late tasks.
      - Robust normalization: IQR-based with sign-preserving centering + hard clipping [-5,5] for all normalized terms.
      - All divisions, logs, and exponents eps-protected; nan/inf replaced deterministically; shape-(N,) guaranteed.
    """
    eps = 1e-08
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

    # Robust IQR normalization with strict bounds
    def robust_iqr_norm(x):
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1 + eps
        center = np.median(x)
        normed = (x - center) / iqr
        return np.clip(normed, -5.0, 5.0)

    # Task intrinsic duration and relative slack
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    rel_slack = slack / task_min_duration  # Parent 2 core insight

    # Unified urgency: smooth, bounded, slack-sensitive [0,1]
    # Sigmoid gate on negative rel_slack (lateness), softened for positive slack
    urgency_neg = 1.0 / (1.0 + np.exp(-np.clip(-rel_slack, 0.0, 20.0)))  # ~0→1 as slack worsens
    urgency_pos = np.exp(-np.clip(rel_slack, 0.0, 20.0) * 0.5)  # decays gently for slack surplus
    deadline_urgency = np.where(slack <= 0, urgency_neg, urgency_pos)

    # Risk-adaptive exponent: depends on slack *and* remaining_work to balance large/small workflows
    # Base penalty increases with lateness, but attenuated for tiny remaining_work (low impact)
    base_risk = np.maximum(0.0, -slack) / (np.median(task_min_duration) + eps)
    rw_ratio = np.clip(remaining_work / (np.median(remaining_work) + eps), 0.1, 10.0)
    risk_exponent = np.clip(1.0 + 0.5 * base_risk * (1.0 + 0.3 * (rw_ratio - 1.0)), 1.0, 3.0)

    # Uncertainty-weighted energy with risk exponent
    energy_risk_weighted = min_incremental_energy * np.power(1.0 + uncertainty, risk_exponent)
    energy_safe = np.maximum(energy_risk_weighted, eps)

    # Criticality-efficiency ratio: upward_rank per unit risk-adjusted energy
    crit_eff_ratio = upward_rank / energy_safe
    crit_eff_ratio = np.clip(crit_eff_ratio, 1e-6, 1e6)
    norm_crit_eff = robust_iqr_norm(crit_eff_ratio)

    # Work-aware energy scaling: only apply when critical *and* deadline-safe
    median_ur = np.median(upward_rank) + eps
    ur_ratio = upward_rank / median_ur
    energy_work_weight = np.where(
        (ur_ratio > 1.5) & (slack > 0),
        np.clip(rw_ratio, 1.0, 2.0),
        1.0
    )
    energy_scaled = min_incremental_energy * energy_work_weight
    norm_energy = robust_iqr_norm(energy_scaled)

    # Starvation control: work-normalized wait pressure, gated by slack>0 and low latency
    work_normalized_wait = ready_wait_time / (remaining_work + eps)
    wait_threshold = np.quantile(work_normalized_wait, 0.9, method='midpoint') + eps
    wait_pressure = np.clip(work_normalized_wait / (wait_threshold + eps), 0.0, 1.0)
    latency_gate = (task_min_duration <= np.median(task_min_duration)).astype(float)
    slack_gate = (slack > 0).astype(float)
    starvation_term = 1.0 - wait_pressure * latency_gate * slack_gate

    # Uncertainty-normalized duration penalty (not energy reversal)
    unc_duration = task_min_duration * (1.0 + uncertainty)
    norm_unc_duration = robust_iqr_norm(unc_duration)

    # Remaining work and uncertainty normalized
    norm_work = robust_iqr_norm(remaining_work)
    norm_unc = robust_iqr_norm(uncertainty)

    # Final score: weighted linear combination; lower = better
    # Weights sum to 1.0 and prioritize urgency, criticality-efficiency, and fairness
    score = (
        0.38 * deadline_urgency +
        0.25 * (1.0 - norm_crit_eff) +  # higher crit_eff → lower score
        0.14 * norm_unc_duration +
        0.09 * norm_energy +
        0.07 * norm_work +
        0.04 * norm_unc +
        0.03 * (1.0 - starvation_term)
    )

    # Final sanitization: ensure finite, bounded, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)

    return score
