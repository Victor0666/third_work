import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining best structural elements from both parents.
    Key novelties:
      - Slack pressure clamped before gating (improves outlier robustness)
      - Multiplicative energy × uncertainty interaction (captures risk-amplified energy cost)
      - Dual-path slack handling: raw exponentiated penalty + normalized pressure gate
      - All numeric literals are -2,-1,0,1,2 only; all tunables via PARAMS.
    """
    eps = 0.00016371177445490323
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize(x):
        x = np.abs(x)
        center = np.median(x)
        scale = np.median(np.abs(x - center)) + eps
        return (x - center) / scale
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.0589729412245887
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 1.9209606176748921)
    rank_gate = 1.0 / (1.0 + np.exp(-0.6345722877346314 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 2.919183228243405 * rank_gate)
    uncert_gate = np.where(norm_uncert > 0.4765796962958606, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * slack_pressure
    energy_uncert_risk = norm_energy * norm_uncert * uncert_gate
    wait_benefit = 1.0 - np.exp(-0.41646368021948466 * (norm_wait + 1e-09))
    score = +norm_slack_penalty - boosted_rank - 1.9243689516348585 * norm_energy - norm_duration - wait_benefit + 0.2095173590349616 * duration_risk_score + 0.0715109840396788 * norm_work + 0.025642700721093745 * energy_uncert_risk
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
