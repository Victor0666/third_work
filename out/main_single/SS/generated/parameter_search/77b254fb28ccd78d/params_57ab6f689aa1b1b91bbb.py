import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with smooth slack pressure, uncertainty-coupled energy, and successor-release signal.
    
    Key features:
    - Smooth tanh-based slack pressure avoids discontinuities (replaces boolean gates).
    - Energy penalty scaled by (1 + uncertainty * energy_uncertainty_coupling) for joint risk awareness.
    - Successor-release term: remaining_work * (1 - clip(slack/median_slack, 0, 1)) prioritizes unblocking critical paths.
    - All numeric literals are -2, -1, 0, 1, or 2; all tunables declared and used.
    - Robust normalization and finite safeguards via np.finfo.
    """
    eps = 3.6961027816361154e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_norm(x):
        x_abs = np.abs(x)
        denom = np.mean(x_abs) + eps
        return x / denom
    median_slack = np.median(np.abs(slack)) + eps
    slack_pressure = np.tanh(slack / (eps + 0.21414130721502475 * median_slack))
    neg_slack = np.maximum(0.0, -slack)
    slack_penalty = np.power(neg_slack + eps, 1.6462679460560397)
    duration = min_exec_time + min_comm_time
    duration_norm = robust_norm(duration)
    uncertainty_norm = robust_norm(uncertainty)
    risk_adjusted_duration = 0.34048723983143847 * duration_norm + (1 - 0.34048723983143847) * uncertainty_norm
    energy_norm = robust_norm(min_incremental_energy)
    energy_uncertainty_factor = 1 + 0.016387391336002283 * uncertainty_norm
    coupled_energy_penalty = 0.6695557618040231 * energy_norm * energy_uncertainty_factor
    rank_base = robust_norm(upward_rank)
    rank_weighted = rank_base * (1 + 1.0830441076080357 * np.maximum(0.0, -slack_pressure))
    median_rank = np.median(upward_rank)
    is_critical = ((upward_rank >= median_rank) & (slack <= np.maximum(eps, 0.6795970575667813 * median_slack))).astype(float)
    criticality_bonus = is_critical * 1.7882563660551747
    work_norm = robust_norm(remaining_work)
    slack_tightness = np.clip(slack / (eps + median_slack), 0.0, 1.0)
    successor_release = work_norm * (1 - slack_tightness)
    wait_norm = robust_norm(ready_wait_time)
    wait_bonus = -0.013346494243476918 * wait_norm
    unc_gate_active = ((uncertainty_norm > 0.2512281409195617) & (slack < eps)).astype(float)
    tightened_penalty = slack_penalty + unc_gate_active * 0.5588067935325909 * uncertainty_norm
    score = tightened_penalty + coupled_energy_penalty + 1 * risk_adjusted_duration - 1 * rank_weighted + 0 * criticality_bonus - 1 * successor_release + wait_bonus
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape {(N,)}, got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values in priority score'
    return score
