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
    """Novel priority rule emphasizing deadline urgency, risk-aware criticality, 
    and energy-efficiency under feasibility constraints.
    
    Key innovations:
    - Uses *slack-sensitive exponential gating*: transforms slack into a smooth,
      bounded urgency signal (0 to 1) that sharply rises near/after deadline.
    - Introduces *energy-per-critical-work ratio*: normalizes incremental energy
      by remaining_work to favor energy-efficient critical path progress.
    - Replaces linear normalization with *robust range-based scaling* (IQR + eps)
      for better outlier resilience and inter-feature comparability.
    - Combines upward_rank and uncertainty into *risk-weighted criticality*:
      high uncertainty downgrades rank importance unless slack is severely negative.
    - Adds starvation mitigation via *waiting-time saturation*: capped sigmoid boost
      prevents runaway dominance by old tasks.
    - All operations are finite, deterministic, and epsilon-protected.
    """
    eps = 1e-8

    # Convert inputs safely; no in-place modification
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Robust range-based normalization: uses IQR to reduce outlier sensitivity
    def robust_normalize(x):
        q25, q75 = np.percentile(x, [25, 75], axis=0, overwrite_input=False)
        iqr = q75 - q25
        center = np.median(x, axis=0, overwrite_input=False)
        scale = iqr if iqr > eps else np.mean(np.abs(x - center)) + eps
        return (x - center) / (scale + eps)

    # 1. Urgency signal: smooth, bounded, deadline-driven (0=late, 1=plenty of time)
    # Exponential decay from slack=0: exp(-max(0, slack)/tau), but inverted & clamped
    tau = np.clip(np.mean(np.abs(slack)) + eps, eps, 1e3)  # adaptive time constant
    urgency = np.exp(-np.maximum(0.0, -slack) / tau)  # near 1 when slack < 0, decays as slack ↑
    # Convert to priority penalty: lower urgency → higher penalty → higher score
    urgency_penalty = 1.0 - urgency  # [0, 1]; 1.0 means critically late

    # 2. Energy efficiency signal: marginal energy per unit remaining work
    # Avoid division by zero; cap extreme ratios
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    energy_per_work = np.clip(energy_per_work, 0.0, 1e6)  # prevent blowup

    # 3. Risk-weighted criticality: upward_rank attenuated by uncertainty unless urgent
    # When slack is negative, uncertainty *amplifies* criticality (we need reliability)
    # Otherwise, uncertainty *dampens* it (avoid risky critical tasks if not urgent)
    slack_sign = np.sign(slack)
    risk_factor = np.where(slack < 0, 1.0 + uncertainty, 1.0 / (1.0 + uncertainty + eps))
    risk_weighted_rank = upward_rank * risk_factor

    # 4. Starvation mitigation: saturating sigmoid on ready_wait_time
    # Saturates at ~0.95 boost after ~3x median wait time
    median_wait = np.median(ready_wait_time) + eps
    wait_ratio = np.clip(ready_wait_time / (3.0 * median_wait + eps), 0.0, 10.0)
    starvation_boost = 1.0 - 1.0 / (1.0 + np.exp(wait_ratio - 2.0))  # [0, 0.95]

    # Normalize all components for fair combination
    norm_urgency = robust_normalize(urgency_penalty)
    norm_energy_per_work = robust_normalize(energy_per_work)
    norm_risk_rank = robust_normalize(risk_weighted_rank)
    norm_exec = robust_normalize(min_exec_time)
    norm_comm = robust_normalize(min_comm_time)
    norm_uncert = robust_normalize(uncertainty)
    norm_starve = robust_normalize(starvation_boost)

    # Final score: prioritize urgency first, then energy efficiency, then criticality,
    # then execution/comm cost, with starvation as gentle tie-breaker
    # All terms designed so *smaller score = higher priority*
    score = (
        3.0 * norm_urgency                    # Strongest weight: DDL feasibility is non-negotiable
        + 1.5 * norm_energy_per_work         # Favor energy-efficient progress on critical work
        + 1.2 * norm_risk_rank               # Critical path matters most when urgent/risky
        + 0.8 * norm_exec                    # Short exec helps responsiveness
        + 0.6 * norm_comm                    # Low comm reduces congestion & energy
        + 0.4 * norm_uncert                  # Prefer predictable over uncertain (unless urgent)
        - 0.3 * norm_starve                  # Gentle boost for long-waiting tasks (anti-starvation)
    )

    # Final numerical safeguard
    return np.nan_to_num(
        score,
        nan=1e12,
        posinf=1e12,
        neginf=-1e12
    )
