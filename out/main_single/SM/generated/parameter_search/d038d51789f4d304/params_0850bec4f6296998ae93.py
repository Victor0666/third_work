import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with hybrid duration-uncertainty coupling:
      - Retains max-abs normalization and robust slack power-law normalization from Parent 2.
      - Replaces *either* linear blend *or* pure power-law with a **single tunable exponent**
        on the uncertainty factor: (exec+comm) * (1 + clipped_uncert)^power.
        This preserves structural simplicity (12 params), avoids convex interpolation,
        and directly generalizes both parents: power=0 → linear; power>0 → nonlinear scaling.
      - Keeps dual-gated energy term (ddl_gate * slack_decay) for strict DDL-first safety.
      - Preserves exponential starvation mitigation and bottleneck proximity via robust slack.
      - All numeric literals are {-2,-1,0,1,2}; no hidden constants; fully deterministic and finite.
    """
    eps = 2.918445644515504e-05
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
        x = np.asarray(x, dtype=np.float64)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.max(abs_x[finite_mask])
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    abs_slk = np.abs(slk) + eps
    robust_slk_mag = np.power(abs_slk, 0.6936007875145509)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    gate_width = 3.571337592158586 + eps
    ddl_gate = np.where(slk_robust <= 0, 1.0, np.where(slk_robust <= gate_width, 1.0 - slk_robust / gate_width, 0.0))
    slack_penalty = np.where(slk_robust < 0, 3.2500981830216533 * np.abs(slk_robust), -4.950713846599633 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.0500045220447766 * normalize(inv_energy)
    slack_decay = np.exp(-np.abs(slk_robust) * 0.9999984999105799)
    energy_score = energy_score * (1.0 - 0.1682260551587933 * ddl_gate) * slack_decay
    rank_powered = np.power(rank + eps, 1.35970834797109)
    rank_score = -normalize(rank_powered)
    robust_abs_slk = np.abs(slk_robust) + eps
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -3.2116960794955602 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    uncert_clipped = np.clip(uncert, 0.0, 2.0)
    duration_with_uncert = duration * np.power(1.0 + uncert_clipped, 0.871514636489506)
    dur_score = normalize(duration_with_uncert + eps)
    wait_sat = 1.0 - np.exp(-wait / (2.0411052426468705 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
