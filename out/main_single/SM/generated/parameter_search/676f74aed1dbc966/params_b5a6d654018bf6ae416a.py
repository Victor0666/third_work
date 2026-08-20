import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with hard DDL-protection gating and simplified robustness.
    
    Key structural improvements:
      - Replaces soft bottleneck gating with *hard slack < 0 activation* for criticality and interaction:
        upward_rank only contributes when slack < 0 — strictly enforces DDL-first scheduling discipline.
      - Removes remaining_work entirely per reflection: no cross-seed benefit, adds noise, violates minimalism.
      - Reverts to mean-abs normalization (more stable under workload shifts than median) and removes fragile clipping/gating logic.
      - Simplifies uncertainty-slack interaction to pure slack-sign masking (no abs(slack_norm) scaling), improving monotonicity.
      - All components are additive, bounded, and use only safe arithmetic; no redundant features or nonlinear distortion.
    """
    eps = 0.0002787434319761896
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x):
        x = np.asarray(x)
        abs_x = np.abs(x)
        scale = np.mean(abs_x) if np.all(np.isfinite(abs_x)) and np.mean(abs_x) > eps else eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    slack_penalty = np.where(slk < 0, 7.223081840466914 * np.abs(slack_norm), -0.242069147527458 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.11333717337062679 * normalize(inv_energy)
    rank_active = np.where(slk < 0, rank, 0.0)
    rank_score = -2.715683262263901 * normalize(rank_active + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.9070963729038961 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.12026422720454144 * wait)
    wait_score = -normalize(wait_sat + eps)
    unc_slack_mask = np.where(slk < 0, 1.0, 0.0)
    unc_slack_interaction = 3.7879459280924603 * uncert_norm * unc_slack_mask
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 362057.84672663274
    min_safe = -finfo.max / 362057.84672663274
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
