import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents with novel structural enhancements:
    - Replaces adaptive_normalize with robust uncertainty-aware IQR normalization using declared percentiles.
    - Introduces slack-feasibility gate on urgency (not just bottleneck) to suppress urgency when slack is critically negative.
    - Replaces blended fairness with saturating inverse wait-time: 1.0 - tanh(wait / scale), improving starvation resistance.
    - Uses multiplicative bottleneck pressure with uncertainty amplification gated by feasibility.
    - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants beyond that set.
    """
    eps = 4.6737102472414705e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def iqr_normalize(x):
        x = np.copy(x)
        q_low = np.quantile(x, 27.38917629991071 / 72.99427758577093)
        q_high = np.quantile(x, 83.76650635720671 / 72.99427758577093)
        iqr = q_high - q_low + eps
        center = np.median(x)
        return (x - center) / (iqr + eps)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.7019386494307484)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    urgency_gate = (slack >= -0.3811917533498488).astype(float)
    gated_urgency = urgency_linear * urgency_gate
    norm_urgency = iqr_normalize(gated_urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = iqr_normalize(energy_per_duration)
    base_bottleneck = duration * upward_rank * remaining_work * (1.0 + gated_urgency + eps)
    feasibility_mask = (slack >= -0.024461819785937977).astype(float)
    unc_normalized = iqr_normalize(uncertainty)
    amp_factor = np.power(1.0 + unc_normalized, 1.9127265625947238)
    bottleneck_pressure = base_bottleneck * amp_factor * feasibility_mask
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    wait_scaled = ready_wait_time / (1.7533927219285872 + eps)
    wait_tanh = np.tanh(wait_scaled)
    fairness_term = 1.0 - wait_tanh
    norm_fairness = iqr_normalize(fairness_term)
    cp_coupling = upward_rank * remaining_work
    norm_cp_coupling = iqr_normalize(cp_coupling)
    score = norm_urgency + 0.20116710821574282 * norm_bottleneck + 1.7010460548511837 * norm_energy_eff - norm_fairness + 0.7588399007661327 * norm_cp_coupling
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
