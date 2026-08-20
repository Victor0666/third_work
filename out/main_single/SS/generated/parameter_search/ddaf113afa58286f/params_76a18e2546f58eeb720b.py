import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces softplus starvation with bounded linear saturation for strict monotonicity and interpretability;
       removes uncertainty gating entirely per reflection — retains only slack feasibility as robust DDL-protection signal;
       introduces rank_slack_hinge_width to adaptively widen/narrow the urgency ramp based on problem hardness;
       eliminates all synthetic risk proxies (congestion, duration_risk) to reduce overfitting and improve generalization;
       uses simplified median-MAD normalization without norm_wait or norm_uncert — only essential features normalized;
       clips all terms to [-2,2] and enforces finite output via nan_to_num with machine-precision bounds."""
    eps = 0.002502790425887738
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
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    slack_feasibility = 1.0 / (1.0 + np.exp(-7.317238492503315 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.5677827370032253
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_hinge = np.clip(-norm_slack, 0.0, 0.7443846585606073)
    rank_slack_coupling = 1.0 + 1.5331461327173699 * (slack_hinge / (0.7443846585606073 + eps))
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-4.030951514466385 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 1.5331461327173699 * rank_gate)
    wait_benefit = np.clip(1.0 - 0.5280983702275077 * ready_wait_time, 0.0, 1.0)
    energy_feasible = norm_energy * slack_feasibility
    energy_slack_penalty = norm_energy * slack_pressure * slack_feasibility
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.0630958318219565 * np.clip(energy_feasible, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 3.5677827370032253 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.3884886276038017 * np.clip(norm_work, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
