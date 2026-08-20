import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: restores rank_slack_coupling for hard deadline feasibility; replaces binary uncertainty gate with smooth sigmoidal activation; retains improved energy-slack interaction and saturating starvation relief."""
    eps = 0.057133802876844365
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
        x_abs = np.abs(x)
        center = np.median(x_abs)
        scale = np.median(np.abs(x_abs - center)) + eps
        return (x_abs - center) / scale
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_pressure = np.power(np.clip(-norm_slack, 0.0, None), 3.1566774195963814)
    rank_gate = np.clip(1.0 - norm_slack / (0.8962044414856878 + eps), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.2234431567609247 * rank_gate)
    uncertainty_activation = 1.0 / (1.0 + np.exp(-1.0002579979005615 * (norm_uncert - 0.39654485683328267)))
    duration_risk_interaction = norm_duration * uncertainty_activation * 1.1136080651144298
    wait_benefit = 1.0 - np.exp(-0.8035683508666486 * (norm_wait + eps))
    energy_slack_penalty = norm_energy * (1.0 + 1.447629541480442 * slack_pressure)
    coupled_rank_reward = norm_rank * (1.0 + 0.26885839813102896 * slack_pressure)
    score = +slack_pressure + energy_slack_penalty + 1.5282966869249375 * norm_energy - coupled_rank_reward - wait_benefit + duration_risk_interaction
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
