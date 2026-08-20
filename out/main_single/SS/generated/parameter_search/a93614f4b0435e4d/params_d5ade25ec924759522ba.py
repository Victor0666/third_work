import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces median-MAD with quantile-based robust normalization;
       introduces congestion-aware gating via (ready_wait_time + uncertainty) threshold;
       replaces boosted_rank with conditional critical-path coupling (upward_rank × remaining_work) activated only under ddl pressure;
       removes duration_robustness and wait_saturation_offset (deemed inactive per diagnostics);
       enforces sign-preserving slack normalization via MAD;
       adds load-successor-release interaction: penalizes high-uncertainty tasks with large remaining_work when slack is tight."""
    eps = 1.2331249878234342e-06
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def quantile_normalize(x):
        x = np.copy(x)
        if N == 1:
            q1 = x[0]
            q3 = x[0]
        else:
            q1 = np.quantile(x, 0.23658743261624432)
            q3 = np.quantile(x, 0.5611474213616869)
        iqr = q3 - q1
        spread = iqr if iqr > eps else eps
        return (x - q1) / spread

    def mad_normalize_slack(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = mad_normalize_slack(slack)
    norm_energy = quantile_normalize(min_incremental_energy)
    norm_duration = quantile_normalize(min_exec_time + min_comm_time)
    norm_rank = quantile_normalize(upward_rank)
    norm_work = quantile_normalize(remaining_work)
    norm_wait = quantile_normalize(ready_wait_time)
    norm_uncert = quantile_normalize(uncertainty)
    ddl_feasible = (slack > 0.0).astype(float)
    ddl_breach = 1.0 - ddl_feasible
    critical_path_coupling = norm_rank * norm_work * ddl_breach
    ddl_protection_gate = 1.0 / (1.0 + np.exp(-5.962133381953219 * slack))
    congestion = ready_wait_time + uncertainty
    congestion_norm = quantile_normalize(congestion)
    congestion_gate = 1.0 / (1.0 + np.exp(-5.962133381953219 * (congestion_norm - 0.33002934701255693)))
    congestion_penalty = norm_duration * congestion_gate * ddl_feasible
    successor_release_penalty = norm_uncert * norm_work * ddl_breach
    wait_benefit = np.clip(0.07203626799475835 * ready_wait_time, 0.0, 2.0)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.4705409032605654
    norm_slack_penalty = quantile_normalize(raw_slack_penalty)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) + np.clip(critical_path_coupling, -2.0, 2.0) - np.clip(norm_rank, -2.0, 2.0) - 0.10420026972293758 * np.clip(norm_energy * ddl_protection_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(congestion_penalty, -2.0, 2.0) + 0.18348374118594646 * np.clip(successor_release_penalty, -2.0, 2.0) + 0.8543243272147536 * np.clip(norm_work * ddl_feasible, -2.0, 2.0) + 1.0011822673189468 * np.clip(norm_slack * ddl_feasible, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
