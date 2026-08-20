import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: adds conditional DDL protection gate, successor-release interaction,
       and load-aware energy modulation — all grounded in counterfactual evidence.
       Uses median-MAD normalization, avoids unstable exponentials, and enforces hard deadline priority."""
    eps = 0.011851254621381162
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
        x = np.asarray(x, dtype=float)
        center = np.median(x) if N > 1 else x[0]
        spread = np.median(np.abs(x - center)) if N > 1 else np.abs(x[0] - center) + eps
        return (x - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_margin_ratio = np.abs(slack) / (min_exec_time + min_comm_time + eps)
    ddl_urgent = np.where((slack <= 0.0) | (slack_margin_ratio < 0.35616118091721116), 1.0, 0.0)
    successor_release_impact = norm_work * norm_rank
    energy_modulation_gate = np.where((norm_uncert > 0.7794662359553223) & (ddl_urgent == 1.0), 1.0, 0.0)
    slack_pressure_norm = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.4060403189561703 * (slack_pressure_norm - 1.0)))
    boosted_rank = norm_rank * (1.0 + 2.176252485750591 * rank_gate * ddl_urgent)
    wait_benefit = 1.0 - np.exp(-0.15135952986511633 * (norm_wait + 4.314085104369535e-07))
    duration_risk_score = norm_duration * norm_uncert * ddl_urgent
    energy_uncert_penalty = norm_energy * norm_uncert * energy_modulation_gate
    score = +norm_slack + norm_slack ** 2.6008670404099603 * ddl_urgent - boosted_rank - successor_release_impact - 1.9250079900485755 * norm_energy * energy_modulation_gate - norm_duration * (1.0 - ddl_urgent) - wait_benefit + 0.35256174967687914 * duration_risk_score + 0.18673447950310662 * energy_uncert_penalty + 1.085144436520048 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
