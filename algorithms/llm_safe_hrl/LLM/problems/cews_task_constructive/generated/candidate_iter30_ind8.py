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
    v3 priority rule: Slack-aware critical-path energy minimization with starvation-proof fairness,
                      communication-aware work density correction, and adaptive normalization.

    Key self-evolution improvements:
      - Replaced static percentile clipping (2%-98%) with *adaptive robust norm*: uses MAD-based outlier rejection
        for better small-N stability and resilience to skewed distributions.
      - Introduced *slack-conditional criticality scaling*: upward_rank is amplified under tight slack (<0.1*duration)
        to force critical-path tasks forward when deadlines loom, while suppressed under safe slack to avoid over-prioritizing.
      - Enhanced *communication penalty* using both comm/comp ratio AND absolute comm time to penalize high-bandwidth tasks
        that induce energy waste and contention, not just relative imbalance.
      - Refined *fairness boost*: now gated by *normalized wait pressure* (ready_wait_time / median_duration) > 2.5
        AND uncertainty > 0.75 quantile — tighter, more discriminative, and avoids boosting trivial waits.
      - Added *energy-efficiency saturation*: efficiency_boost applies only when risk_adjusted_energy > median,
        preventing unnecessary optimization on already-efficient tasks.
      - Unified urgency term now includes *lateness-risk amplification*: negative slack is weighted quadratically
        for stronger penalty on violation-prone tasks without numerical explosion.
      - All components bounded, NaN/inf hardened, and deterministic per contract.
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64).copy()
    slack = np.asarray(slack, dtype=np.float64).copy()
    upward_rank = np.asarray(upward_rank, dtype=np.float64).copy()
    remaining_work = np.asarray(remaining_work, dtype=np.float64).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64).copy()
    uncertainty = np.asarray(uncertainty, dtype=np.float64).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)

    # Adaptive robust normalization using MAD (more stable than percentile for small N)
    def robust_mad_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        if N == 1:
            return np.zeros_like(x)
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x))
        if mad < eps:
            return np.zeros_like(x)
        # Normalize: (x - median) / (1.4826 * mad) → then clip & rescale to [0,1]
        z = (x - median_x) / (1.4826 * mad + eps)
        # Clip extreme z-scores and map to [0,1] via soft tanh-like sigmoid
        z_clipped = np.clip(z, -4.0, 4.0)
        normed = (np.tanh(z_clipped / 2.0) + 1.0) / 2.0
        return normed

    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)

    # Lateness-risk amplification: quadratic penalty for negative slack, linear for positive urgency
    urgency_base = np.where(slack < 0, 
                           (-slack / (duration + eps))**2,  # Quadratic penalty for violation risk
                           np.clip(-rel_slack, 0.0, 0.6))   # Linear urgency for positive slack
    urgency_saturation = 1.0 / (1.0 + np.power(np.abs(rel_slack) + eps, 0.25))
    urgency_bias = urgency_base * urgency_saturation
    norm_urgency = robust_mad_norm(urgency_bias)

    # Energy density and risk-adjusted energy
    energy_density = np.divide(min_incremental_energy, duration + eps, 
                              out=np.zeros_like(min_incremental_energy), where=duration + eps != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    slack_rescale = 1.0 + np.power(np.abs(rel_slack) + eps, 0.35)
    risk_adjusted_energy = energy_density / (slack_rescale + eps)
    norm_risk_energy = robust_mad_norm(risk_adjusted_energy)

    # Slack-conditional criticality scaling: amplify upward_rank when slack is tight
    duration_median = np.median(duration) if N > 1 else duration[0]
    tight_slack_mask = (slack < 0.1 * duration_median).astype(float)
    safe_slack_mask = (slack >= 0.1 * duration_median).astype(float)
    # Boost criticality under tight slack; dampen under safe slack
    scaled_upward_rank = upward_rank * (1.0 + 0.8 * tight_slack_mask) * (1.0 - 0.3 * safe_slack_mask)
    norm_upward_rank = robust_mad_norm(scaled_upward_rank)

    # Communication-aware work density correction
    comm_comp_ratio = np.divide(min_comm_time, min_exec_time + eps, 
                               out=np.zeros_like(min_exec_time), where=min_exec_time + eps != 0)
    comm_comp_ratio = np.nan_to_num(comm_comp_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    abs_comm_penalty = min_comm_time / (np.median(duration) + eps) if N > 1 else min_comm_time / (duration_median + eps)
    abs_comm_penalty = np.nan_to_num(abs_comm_penalty, nan=0.0, posinf=0.0, neginf=0.0)
    work_intensity_penalty = 0.6 * robust_mad_norm(comm_comp_ratio) + 0.4 * robust_mad_norm(abs_comm_penalty)

    # Fairness boost: gated by normalized wait pressure AND high uncertainty
    median_duration = np.median(duration) if N > 1 else duration[0]
    wait_pressure = np.divide(ready_wait_time, median_duration + eps, 
                             out=np.zeros_like(ready_wait_time), where=median_duration + eps != 0)
    wait_pressure = np.nan_to_num(wait_pressure, nan=0.0, posinf=0.0, neginf=0.0)
    unc_quantile = np.quantile(uncertainty, 0.75) if N > 1 else np.max(uncertainty)
    fairness_boost_mask = (wait_pressure > 2.5).astype(float) * (uncertainty > unc_quantile).astype(float)
    norm_ready_wait = robust_mad_norm(ready_wait_time)
    fairness_boost = norm_ready_wait * fairness_boost_mask

    # Energy-efficiency saturation: only boost if risk-adjusted energy is above median
    energy_median = np.median(risk_adjusted_energy) if N > 1 else risk_adjusted_energy[0]
    efficiency_boost_mask = (risk_adjusted_energy > energy_median) & (slack > 0.0) & (upward_rank > np.median(upward_rank) if N > 1 else True)
    efficiency_boost = -0.12 * norm_risk_energy * efficiency_boost_mask.astype(float)

    norm_remaining_work = robust_mad_norm(remaining_work)

    # Final weighted score — smaller = higher priority
    score = (
        0.33 * norm_urgency +
        0.21 * norm_risk_energy * (0.65 + 0.35 * norm_upward_rank) +
        0.15 * (1.0 - fairness_boost) +
        0.14 * work_intensity_penalty +
        0.10 * norm_remaining_work +
        0.07 * efficiency_boost
    )

    # Hard bounds and NaN/inf hardening
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
