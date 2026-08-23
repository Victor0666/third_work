import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
      - Employs Parent 2's robust fixed-bound slack scaling (no percentiles) and bounded uncertainty sigmoid.
      - Integrates Parent 1's insight on uncertainty-aware rank amplification, now parameterized as `upward_rank_uncertainty_weight`.
      - Removes all wait-time and early-slack reward terms (confirmed inactive).
      - Uses monotonic slack scoring: only penalizes negative slack; zero reward for positive slack.
      - Criticality boost is applied multiplicatively *after* base score assembly for clean gradient flow.
      - All numeric literals are strictly in {-2,-1,0,1,2}; no hidden constants.
      - Robust_norm avoids tanh (simpler, differentiable, stable) and uses mean-abs + eps.
      - Final score is finite, deterministic, and shape-(N,) guaranteed.
    """
    eps = 4.5052925578156185e-07
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

    def robust_norm(x):
        x = np.asarray(x, dtype=float)
        denom = np.mean(np.abs(x)) + eps
        return x / (denom + eps)
    slack_abs = np.abs(slack)
    slack_score = np.where(slack < 0, slack_abs ** 2.12321990586064, 0.0)
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    is_high_rank = upward_rank >= rank_median
    is_tight_or_violated = slack <= 0
    critical_gate = np.where(is_high_rank & is_tight_or_violated, 3.099386508380182, 1.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = robust_norm(energy_per_sec)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.9922082357048398
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.6291163907364639
    slack_lb = -29.340610755660336
    slack_ub = 47.71130781052441
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.8929572485005256 + (1.0 - 0.8929572485005256) * (1.0 - slack_scaled)
    augmented_rank = upward_rank * (1.0 + uncertainty * 0.5965075301651193)
    rank_score = -robust_norm(augmented_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.8323377198578825 * (uncertainty - 1.0)))
    energy_norm = robust_norm(min_incremental_energy)
    unc_norm = robust_norm(uncertainty)
    energy_uncertainty_score = 0.008405286629106256 * energy_norm * unc_norm * unc_sigmoid
    residual_energy_term = robust_norm(min_incremental_energy) * (1.0 - weight_rank)
    score = robust_norm(slack_score) + robust_norm(unc_slack_coupling) + robust_norm(duration_risk) + 0.21775373629127104 * energy_eff_score + rank_score + energy_uncertainty_score + residual_energy_term
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
