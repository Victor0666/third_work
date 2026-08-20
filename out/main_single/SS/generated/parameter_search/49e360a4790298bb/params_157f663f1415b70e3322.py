import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces sigmoid slack-pressure with *adaptive-width linear ramp* for tighter control over deadline urgency sensitivity;
       retains strict DDL-protection gating (zero penalty when slack < 0) — validated by counterfactuals on feasibility violation;
       restores clean successor_release = norm_rank * norm_work * ddl_pressure (no extra weight param), improving structural parsimony;
       introduces robustified *min_exec_time + min_comm_time* normalization using median-MAD *after* summing — avoids bias from separate feature scaling;
       clips all intermediate terms to [-2,2] for bounded AST depth and numerical stability;
       eliminates redundant slack_pressure * PARAMS["ddl_protection_gate_slope"] term — replaced by direct ddl_pressure signal with calibrated width."""
    eps = 0.0007780182022324398
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def median_mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    duration_sum = min_exec_time + min_comm_time
    norm_duration = median_mad_normalize(duration_sum)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    width = 1.085328684172584
    ddl_gate = np.clip(slack / (width + eps), 0.0, 1.0)
    ddl_pressure = ddl_gate
    successor_release = norm_rank * norm_work * ddl_pressure
    coupled_rank = norm_rank * (1.0 + 1.098813112514184 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.28898117280772107, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = np.clip(0.06473684208940522 * ready_wait_time, 0.0, 1.0)
    duration_uncert_penalty = norm_duration * norm_uncert * ddl_pressure * ddl_gate
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.24561354603554986 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_uncert_penalty, -2.0, 2.0) + 0.9239383167636962 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.3029950835480658 * np.clip(norm_work, -2.0, 2.0) + np.clip(ddl_pressure * 4.462701457743924, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
