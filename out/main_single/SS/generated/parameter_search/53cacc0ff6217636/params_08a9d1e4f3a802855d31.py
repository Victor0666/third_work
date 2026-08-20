import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: combines Parent 2's lateness multiplier with Parent 1's robustness logic;
       adds novel work-slack coupling to prioritize high-computation tasks under deadline pressure;
       replaces fragile norm_slack clipping with monotonic slack-pressure-based interactions;
       uses MAD-only normalization without absolute-value clipping to preserve rank ordering fidelity;
       enforces hard-deadline compliance via selective lateness scaling of penalties only;
       eliminates redundant gates while retaining smooth uncertainty activation and saturating wait relief."""
    eps = 0.0005199270559377023
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
        if N == 0:
            return np.zeros(0, dtype=float)
        center = np.median(x)
        scale = np.median(np.abs(x - center)) + eps
        return (x - center) / (scale + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_pressure_raw = np.clip(-norm_slack, 0.0, None)
    slack_pressure = np.power(slack_pressure_raw + eps, 1.821328394319924)
    is_late = (slack < -eps).astype(float)
    lateness_multiplier = 1.0 + 0.7716828135797029 * is_late
    rank_gate = np.clip(1.0 - norm_slack / (0.5522479184325941 + eps), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 2.3777620900123164 * rank_gate)
    uncertainty_activation = 1.0 / (1.0 + np.exp(-2.9728602891261895 * (norm_uncert - 0.0)))
    duration_risk_interaction = norm_duration * uncertainty_activation * 0.6894339640601723
    wait_benefit = 1.0 - np.exp(-0.6428778330197281 * (norm_wait + eps))
    energy_slack_penalty = norm_energy * (1.0 + 1.2137570759742435 * slack_pressure)
    coupled_rank_reward = norm_rank * (1.0 + 0.5464779948240566 * slack_pressure)
    work_slack_penalty = norm_work * (1.0 + 0.9009095337815181 * slack_pressure)
    score = +lateness_multiplier * slack_pressure + lateness_multiplier * energy_slack_penalty + lateness_multiplier * 1.4177066315336626 * norm_energy + lateness_multiplier * work_slack_penalty - coupled_rank_reward - wait_benefit + lateness_multiplier * duration_risk_interaction
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape {(N,)}, got {score.shape}'
    return score
