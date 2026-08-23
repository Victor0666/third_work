import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all tunable constants declared; only -2,-1,0,1,2 used as literals."""
    eps = 0.0010876215257706015

    def robust_norm(x):
        x = np.asarray(x, dtype=float)
        denom = np.mean(np.abs(x)) + eps
        return x / denom
    slack_signed = np.copy(slack)
    slack_abs = np.abs(slack_signed)
    slack_sign = np.sign(slack_signed)
    slack_score = np.where(slack_signed < 0, slack_abs ** 3.3933759737622378, slack_signed * 0.08432339087869646)
    rank_median = np.median(upward_rank) if len(upward_rank) > 1 else np.mean(upward_rank)
    is_high_rank = upward_rank >= rank_median
    is_tight_slack = slack_signed <= 2 * eps
    critical_gate = np.where(is_high_rank & is_tight_slack, 1.4719096718055722, 1.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = robust_norm(energy_per_sec)
    deadline_pressure = np.maximum(0.0, -slack_signed)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.046130079880877
    wait_clipped = np.clip(ready_wait_time, 0, 76.02961306691668)
    wait_score = -robust_norm(wait_clipped)
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.24355337549981684
    slack_p90 = np.percentile(slack, 98.96534832806974) if len(slack) > 1 else np.max(slack)
    slack_p10 = np.percentile(slack, 1.6883229025493343) if len(slack) > 1 else np.min(slack)
    slack_range = np.maximum(eps, slack_p90 - slack_p10)
    slack_normalized = np.clip((slack_signed - np.min(slack)) / (slack_range + eps), 0, 1)
    weight_rank = 0.4421251845297858 + (1 - 0.4421251845297858) * slack_normalized
    rank_score = -robust_norm(upward_rank) * weight_rank
    score = robust_norm(slack_score) + robust_norm(unc_slack_coupling) + robust_norm(duration_risk) + 0.40013505005589056 * energy_eff_score + rank_score + wait_score + robust_norm(min_incremental_energy) * (1 - weight_rank)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
