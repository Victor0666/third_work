import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces sigmoid gates with bounded linear interpolation;
       uses sign-preserving MAD normalization; removes inactive parameters (duration_robustness);
       introduces critical successor release interaction via upward_rank * remaining_work under ddl_gate;
       applies clipped linear DDL-protection gate instead of soft sigmoid for stronger hard-constraint enforcement;
       prioritizes tasks with smallest slack first, then critical path impact, then starvation relief."""
    eps = 0.0018589159325927906
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = np.maximum(mad, eps)
        return (x - med) / spread
    norm_slack = mad_normalize(slack)
    norm_energy = mad_normalize(min_incremental_energy)
    norm_duration = mad_normalize(min_exec_time + min_comm_time)
    norm_rank = mad_normalize(upward_rank)
    norm_work = mad_normalize(remaining_work)
    norm_wait = mad_normalize(ready_wait_time)
    norm_uncert = mad_normalize(uncertainty)
    ddl_gate = 0.7948364912645165 * (1.0 + np.tanh(3.258037314289016 * slack))
    critical_path_leverage = norm_rank * norm_work * ddl_gate
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.9818201248539409
    norm_slack_penalty = mad_normalize(raw_slack_penalty)
    pressure = np.clip(1.0 + 1.6877993857108693 * -norm_slack, 0.0, 2.0)
    uncert_gate = (norm_uncert >= 0.044244634738884014).astype(float)
    wait_benefit = np.clip(1.0 - np.exp(-0.07951353277659423 * ready_wait_time), 0.0, 0.9045251144297267)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(norm_rank * pressure, -2.0, 2.0) - 1.558479463410887 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.11141612344113422 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.8842031287188931 * np.clip(norm_work, -2.0, 2.0) + 0.6937088042545946 * np.clip(norm_rank * norm_work * (1.0 - ddl_gate), -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
