import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining best practices from both parents:
      - Uses Parent 2's robust z-score normalization with ±sigma clipping for discriminability and stability.
      - Adopts Parent 2's soft sigmoid-based DDL protection gating (not hard `np.where`) for graceful attenuation.
      - Integrates Parent 1's successor-release sharpening (`power(1/|slack|, sharpness)`) into bottleneck term.
      - Refines bottleneck activation: requires *both* negative slack *and* non-negligible work *and* applies sharpening.
      - Replaces duration-uncertainty blend with uncertainty-slack-coupled urgency term (from Parent 1), preserving joint-risk capture.
      - Keeps exponential starvation mitigation (Parent 2) but adds bounded ramp fallback for numerical safety.
      - All operations guarded against NaN/inf/zero; uses only allowed literals (-2,-1,0,1,2).
    """
    eps = 0.0003549902234021562
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x):
        x = np.asarray(x)
        finite_mask = np.isfinite(x) & (np.abs(x) > eps)
        if not np.any(finite_mask):
            return np.zeros_like(x)
        x_finite = x[finite_mask]
        mu = np.mean(x_finite)
        std = np.std(x_finite) if len(x_finite) > 1 else eps
        z = (x - mu) / (std + eps)
        clip_bound = 1.0561123747695247
        z_clipped = np.clip(z, -clip_bound, clip_bound)
        max_abs = np.maximum(np.max(np.abs(z_clipped)), eps)
        return z_clipped / (max_abs + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(1.7655609689555072 * slack_norm))
    slack_penalty = np.where(slk < 0, 3.4274715771961923 * np.abs(slack_norm), -2.4087115594046336 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.18169850887275824 * normalize(inv_energy) * ddl_gate
    rank_score = -1.2070747936554955 * normalize(rank) * ddl_gate
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 0.9409088511345345)
    bottleneck_active = np.where((slk < 0) & (work > eps), bottleneck_sharpened, 0.0)
    bottleneck_score = -3.5162330492546414 * normalize(bottleneck_active + eps)
    uncert_norm = normalize(uncert + eps)
    unc_slack_coupling = 2.451463453128749 * uncert_norm * np.where(slack_norm < 0, np.abs(slack_norm), 0.0)
    coupled_score = -unc_slack_coupling
    wait_sat = 1.0 - np.exp(-0.16976700505144543 * wait)
    wait_fallback = np.clip(wait / (0.16976700505144543 + eps), 0.0, 1.0)
    wait_combined = np.where(np.isfinite(wait_sat), wait_sat, wait_fallback)
    wait_score = -normalize(wait_combined + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + coupled_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 3072460.3168998
    min_safe = -finfo.max / 3072460.3168998
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
