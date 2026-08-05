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
    Self-evolved priority rule v4: Hybrid quantile-dynamic urgency with critical-path energy coupling,
    starvation-aware work-weighted wait pressure, and robust multi-scale normalization.

    Key innovations:
      - Dual-threshold slack gating: uses q20 for *early intervention* and q50 for *critical amplification*,
        enabling finer-grained deadline hardness control than v1/v2.
      - Critical-path energy coupling refined via *upward_rank × (1 + normalized_slack_risk)* scaled by
        task_duration / (min_incremental_energy + eps), preserving HEFT semantics while embedding risk.
      - Starvation relief upgraded to *work-and-slack-significance weighted wait pressure*: 
        (ready_wait_time / task_duration) × sigmoid(remaining_work / median_rw) × sigmoid(-slack / median_dur),
        ensuring boost activates only when both computation weight and urgency deficit co-occur.
      - Robust normalization uses *adaptive scaling*: IQR for N>=5, MAD for 2<=N<5, identity for N==1 — 
        balancing stability and sensitivity across all cardinalities.
      - Energy risk scoring now uses *smooth clipped exponential*: 1 + 0.5 * exp(max(0, -slack)/task_duration) - 1,
        bounded and differentiable, avoiding numeric explosion.
      - Introduces *uncertainty-aware communication penalty*: min_comm_time × (1 + 0.5 * uncertainty),
        scaled by slack violation status to prioritize data-bound tasks under risk.
      - Final weights rebalanced for DDL-hardness dominance (0.62), synergy efficiency (0.23), energy control (0.07),
        starvation relief (0.04), and communication overhead (0.04).
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

    # Safely clip and sanitize inputs
    min_exec_time = np.clip(np.nan_to_num(min_exec_time, nan=eps, posinf=eps, neginf=eps), eps, 1e9)
    min_comm_time = np.clip(np.nan_to_num(min_comm_time, nan=eps, posinf=eps, neginf=eps), eps, 1e9)
    min_incremental_energy = np.clip(np.nan_to_num(min_incremental_energy, nan=eps, posinf=eps, neginf=eps), eps, 1e9)
    slack = np.nan_to_num(slack, nan=0.0, posinf=1e9, neginf=-1e9)
    upward_rank = np.clip(np.nan_to_num(upward_rank, nan=eps, posinf=1e6, neginf=eps), eps, 1e6)
    remaining_work = np.clip(np.nan_to_num(remaining_work, nan=eps, posinf=1e9, neginf=eps), eps, 1e9)
    ready_wait_time = np.clip(np.nan_to_num(ready_wait_time, nan=0.0, posinf=1e6, neginf=0.0), 0.0, 1e6)
    uncertainty = np.clip(np.nan_to_num(uncertainty, nan=0.0, posinf=1e3, neginf=0.0), 0.0, 1e3)

    task_duration = min_exec_time + min_comm_time
    task_duration = np.maximum(task_duration, eps)

    # Compute relative slack with clipping for stability
    rel_slack = np.divide(slack, task_duration, out=np.zeros_like(slack), where=task_duration != 0)
    rel_slack_clipped = np.clip(rel_slack, -10.0, 10.0)

    # Dynamic quantile anchoring: compute robust quantiles
    if N > 1:
        q10, q20, q30, q50, q70, q90 = np.quantile(
            rel_slack_clipped, [0.1, 0.2, 0.3, 0.5, 0.7, 0.9], method='midpoint'
        )
    else:
        q10 = q20 = q30 = q50 = q70 = q90 = rel_slack_clipped[0]

    # Urgency penalty: dual-threshold with smooth transition
    violated_mask = slack < 0
    tight_mask = ~violated_mask & (rel_slack < q20)
    mid_mask = ~violated_mask & (rel_slack >= q20) & (rel_slack < q50)
    relaxed_mask = ~violated_mask & (rel_slack >= q50)

    urgency_penalty = np.zeros_like(slack)
    urgency_penalty[violated_mask] = np.clip(
        -slack[violated_mask] / (task_duration[violated_mask] + eps), 0.0, 10.0
    )
    urgency_penalty[tight_mask] = np.clip(q50 - rel_slack[tight_mask], 0.0, 4.0)
    urgency_penalty[mid_mask] = np.clip(q50 - rel_slack[mid_mask], 0.0, 2.0)
    urgency_penalty[relaxed_mask] = np.clip(q90 - rel_slack[relaxed_mask], 0.0, 0.5)

    # Slack-aware critical-path coupling: upward_rank × (1 + normalized risk)
    slack_risk = np.maximum(0.0, -slack) / (task_duration + eps)
    normalized_slack_risk = np.clip(slack_risk, 0.0, 5.0)
    coupling_factor = 1.0 + 0.8 * normalized_slack_risk
    base_synergy = upward_rank * coupling_factor * task_duration / (min_incremental_energy + eps)
    base_synergy = np.clip(base_synergy, eps, 1e8)

    # Communication overhead penalty: uncertainty-aware and violation-sensitive
    comm_penalty = min_comm_time * (1.0 + 0.5 * uncertainty)
    comm_penalty = np.where(violated_mask, comm_penalty * 4.0,
                           np.where(tight_mask, comm_penalty * 1.5, comm_penalty * 0.3))

    # Smooth clipped exponential energy risk scaling
    rel_slack_distance = np.maximum(0.0, -slack) / (task_duration + eps)
    risk_weighted_energy = min_incremental_energy * (
        1.0 + 0.5 * (np.exp(np.clip(rel_slack_distance, 0.0, 5.0)) - 1.0)
    )
    risk_weighted_energy = np.clip(risk_weighted_energy, eps, 1e9)

    # Starvation relief: work-and-slack-significance weighted wait pressure
    median_task_dur = np.median(task_duration) + eps
    median_rw = np.median(remaining_work) + eps
    wait_ratio = np.clip(ready_wait_time / median_task_dur, 0.0, 5.0)
    rw_sigmoid = 1.0 / (1.0 + np.exp(-(remaining_work / median_rw - 0.5) / 0.2))
    slack_deficit_sigmoid = 1.0 / (1.0 + np.exp((slack + 0.5 * median_task_dur) / (task_duration + eps)))
    wait_boost = wait_ratio * rw_sigmoid * slack_deficit_sigmoid
    wait_boost = np.clip(wait_boost, 0.0, 1.5)

    # Adaptive robust normalization function
    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e10, 1e10)
        if N == 1:
            return np.array([0.0])
        elif N >= 5:
            q25 = np.quantile(x, 0.25, method='midpoint')
            q75 = np.quantile(x, 0.75, method='midpoint')
            iqr = q75 - q25 + eps
            center = np.median(x)
            z = (x - center) / iqr
            return np.clip(z, -10.0, 10.0)
        else:
            center = np.median(x)
            abs_devs = np.abs(x - center)
            mad = np.median(abs_devs) + eps
            z = (x - center) / mad
            return np.clip(z, -6.0, 6.0)

    norm_urgency = robust_normalize(urgency_penalty)
    norm_synergy = robust_normalize(base_synergy)
    norm_energy = robust_normalize(risk_weighted_energy)
    norm_comm = robust_normalize(comm_penalty)
    norm_wait = robust_normalize(wait_boost)

    # Final weighted score: smaller is better
    score = (
        0.62 * norm_urgency
        - 0.23 * norm_synergy
        + 0.07 * norm_energy
        + 0.04 * norm_comm
        + 0.04 * norm_wait
    )

    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    score = np.clip(score, -1e10, 1e10)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
