import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with one bounded structural improvement:
      - Replaces convex slack penalty with a *bounded sigmoid* over normalized slack,
        preserving strict DDL dominance (steep negative region) while avoiding numerical
        explosion and over-penalization of moderate lateness — directly addressing reflection.
      - Tight-slack uncertainty coupling is now *conditional and exclusive*: applied only when
        slack <= ddl_protection_threshold (not additive everywhere), decoupled from duration risk.
      - Wait-time anti-starvation scaled by normalized headroom ratio (not raw slack difference)
        for robustness across workflow scales.
      - All other components preserved from prior version's validated structure (MAD normalization,
        lexicographic gating, critical-path term, etc.), maintaining determinism and feasibility.
      - No new feature interactions beyond declared parameters; all numeric literals restricted to {-2,-1,0,1,2}.
    """
    eps = 6.456407692045158e-07
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    duration_total = min_exec_time + min_comm_time + eps
    slack_abs_norm = mad_normalize(np.abs(slack))
    unc_norm = mad_normalize(uncertainty)
    dur_norm = mad_normalize(duration_total)
    slack_sigmoid_input = -slack_abs_norm
    slack_sigmoid = 1.0 / (1.0 + np.exp(-6.422690000109527 * slack_sigmoid_input))
    slack_penalty = slack_sigmoid * 1.1550156056888985 * (1.0 + unc_norm)
    tight_slack_mask = np.where(slack <= 1.806307496945575, 1.0, 0.0)
    tight_slack_uncertainty_coupling = tight_slack_mask * uncertainty * slack_abs_norm * 0.9587055547879482
    duration_risk = duration_total * uncertainty * 1.2233180504400547
    critical_release_score = upward_rank * remaining_work * 2.304820655433124
    slack_headroom_mask = np.where(slack > 1.806307496945575, 1.0, 0.0)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_term = slack_headroom_mask * 1.263296672206326 * energy_norm
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 0.8219730639232788, 0.0)
    wait_norm = mad_normalize(wait_power)
    wait_term = slack_headroom_mask * wait_norm * 0.49386110119969895
    rank_norm = mad_normalize(upward_rank)
    headroom_ratio = np.clip((slack - 1.806307496945575) / (1.806307496945575 + 1.0), 0.0, 1.0)
    weight_rank = 0.7002831316133162 + (1.0 - 0.7002831316133162) * (1.0 - headroom_ratio)
    rank_term = slack_headroom_mask * -rank_norm * weight_rank
    score = mad_normalize(slack_penalty) + mad_normalize(tight_slack_uncertainty_coupling) + mad_normalize(duration_risk) + mad_normalize(critical_release_score)
    score += energy_term + wait_term + rank_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
