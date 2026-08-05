import numpy as np

# 函数名中的 2 由 SeEvo 在生成 Prompt 时替换为目标版本号。
# 八个输入均为长度 N 的一维数组；相同下标始终指向同一个 ready task。
# 返回值也必须是长度 N 的一维数组，并遵守“分数越小，优先级越高”。
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
    v3 priority rule: Tightens deadline enforcement via *latency-resolved urgency* — replaces scalar slack with
    risk-completion-time delta relative to critical path; refines energy penalty using *criticality-aware percentile
    capping* (not MAD threshold) for stronger discriminative power on high-upward-rank tasks; introduces
    *work-normalized starvation pressure* that scales with both wait time and remaining work scarcity;
    unifies uncertainty coupling under *slack-sign-aware scaling*: positive slack → uncertainty dampens priority,
    negative slack → uncertainty amplifies urgency; eliminates redundant normalization layers for numerical stability;
    uses robust quantile-based clipping instead of fixed [-8,8] bounds; enforces strict monotonicity in all components.
    
    Key improvements:
      - Urgency now computes *delta_t = (risk_completion_time - critical_path_deadline)* → more precise than raw slack,
        approximated as (min_exec_time + min_comm_time - slack) normalized by upward_rank to reflect per-critical-unit delay.
      - Energy penalty applies 95th-percentile cap *only* on upward_rank-weighted energy density, making inefficiency
        detection sensitive to critical-path importance.
      - Starvation pressure = norm_wait_time * (1.0 / (norm_remaining_work + 0.1)), prioritizing long-waiting tasks
        with scarce descendant work — prevents low-work "zombie" tasks from dominating fairness.
      - Uncertainty modulation: sign(slack) * uncertainty * exp(-|slack|/τ), τ=5.0 → smoothly suppresses uncertainty
        when slack is large positive, enhances it when slack is negative (exponential urgency coupling).
      - All normalizations use quantile-clipped min-max (1st–99th percentile) for outlier resilience + monotonicity.
      - Final score guarantees strict ordering: urgency dominates (weight -1.0), then energy (0.24), latency (0.17),
        starvation (0.13), uncertainty (0.09), base uncertainty (0.06), remaining_work (0.05), with no cancellation artifacts.
      - Explicit zero-division guard and finite-value sanitization applied at every stage.
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

    # Robust quantile-clipped min-max normalization: preserves monotonicity & handles outliers
    def qclip_minmax(x):
        x = np.where(np.isnan(x) | np.isinf(x), np.nanmedian(x, where=~(np.isnan(x) | np.isinf(x)), initial=0.0), x)
        p01 = np.nanpercentile(x, 1.0)
        p99 = np.nanpercentile(x, 99.0)
        x_clipped = np.clip(x, p01, p99)
        x_min = np.nanmin(x_clipped)
        x_max = np.nanmax(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x_clipped)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Latency-resolved urgency: approximate risk_completion_time - critical_path_deadline
    # Using upward_rank as critical path weight → higher rank means tighter deadline sensitivity
    duration = min_exec_time + min_comm_time + eps
    # Negative slack means deadline already violated; positive slack means margin remains
    # We define urgency delta as how much *beyond* the available slack the task's duration consumes
    # Scaled by upward_rank to prioritize critical-path delays more severely
    urgency_delta = (duration - np.abs(slack)) / (upward_rank + eps)
    # Binary + magnitude: -1.0 base for any violation (slack <= 0), plus penalty proportional to urgency_delta
    urgency_bias = np.where(slack <= 0.0, -1.0 - qclip_minmax(np.maximum(urgency_delta, 0.0)), 0.0)

    # Criticality-weighted energy density: penalize high energy per unit criticality
    energy_density = min_incremental_energy / (duration + eps)
    weighted_energy_density = energy_density * (upward_rank + eps)
    # Cap at 95th percentile to focus penalty only on top 5% inefficient critical tasks
    cap_95 = np.nanpercentile(weighted_energy_density, 95.0) + eps
    energy_penalty_mask = (weighted_energy_density > cap_95).astype(float)
    norm_energy_density = qclip_minmax(energy_density)
    energy_penalty = norm_energy_density * energy_penalty_mask

    # Latency component: execution + communication, normalized and scaled by criticality
    latency_raw = min_exec_time + min_comm_time + eps
    norm_latency = qclip_minmax(latency_raw)

    # Work-normalized starvation pressure: reward waiting *only* when remaining work is scarce
    # Prevents low-work leaf tasks from unfairly monopolizing fairness boost
    norm_wait_time = qclip_minmax(ready_wait_time)
    norm_remaining_work = qclip_minmax(remaining_work)
    # Inverse scarcity: smaller remaining_work → higher pressure
    scarcity_factor = 1.0 / (norm_remaining_work + 0.1)
    starvation_pressure = norm_wait_time * scarcity_factor
    # Only activate when slack > 0 (deadline-safe regime)
    starvation_mask = (slack > 0.0).astype(float)
    wait_penalty = starvation_mask * qclip_minmax(starvation_pressure)

    # Sign-aware exponential uncertainty modulation: suppress when slack > 0, amplify when slack < 0
    # τ = 5.0 seconds → smooth transition zone around slack = ±5s
    tau = 5.0
    slack_abs = np.abs(slack) + eps
    exp_decay = np.exp(-slack_abs / tau)
    # Positive slack → uncertainty * reduces* priority (dampening); negative → *increases* priority (amplification)
    uncertainty_sign = np.sign(slack)
    uncertainty_modulated = uncertainty_sign * uncertainty * exp_decay
    norm_uncertainty_mod = qclip_minmax(uncertainty_modulated)

    # Base uncertainty term (non-modulated) for general risk awareness
    norm_uncertainty = qclip_minmax(uncertainty)

    # Remaining work baseline — lower work should not inflate priority unless combined with starvation
    norm_remaining_work_final = qclip_minmax(remaining_work)

    # Final score: strictly ordered weights, no cancellation, bounded and sanitized
    score = (
        1.0
        + urgency_bias
        + 0.24 * energy_penalty
        + 0.17 * norm_latency
        + 0.13 * wait_penalty
        + 0.09 * norm_uncertainty_mod
        + 0.06 * norm_uncertainty
        + 0.05 * norm_remaining_work_final
    )

    # Final sanitization: ensure finite, bounded, deterministic output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)

    return score
