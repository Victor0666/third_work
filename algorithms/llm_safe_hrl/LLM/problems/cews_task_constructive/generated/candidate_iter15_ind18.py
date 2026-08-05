import numpy as np

def get_task_priority_v2(
    min_exec_time,
    min_comm_time,
    min_incremental_energy,
    slack,
    upward_rank,
    remaining_work,
    ready_wait_time,
    uncertainty
):
    """
    v2 evolution: Fixes lexicographic dilution by decoupling urgency normalization from energy terms;
    replaces MAD-based normalization with *rank-based percentile scaling* for stable ordinal preservation;
    removes edge-risk modulation (caused inversions) and replaces with *uncertainty-aware communication penalty*
    only applied to high-slack tasks; introduces *sigmoid urgency* with adaptive steepness based on global slack distribution;
    redefines SEER as *slack-conditioned efficiency ratio* using soft gating (not hard zeroing) to preserve gradient;
    adds *critical-path density fairness* term to prevent starvation of long-latency critical paths;
    all terms weighted to enforce strict priority hierarchy: urgency > SEER > CP-density > fairness > uncertainty.
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Rank-based percentile normalization: preserves ordinal relationships, avoids MAD instability
    def normalize_rank(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        # Use percentile ranks scaled to [-1, 1] for robustness
        sorted_x = np.sort(x)
        ranks = np.searchsorted(sorted_x, x, side='left') / max(len(sorted_x) - 1, 1)
        return 2.0 * ranks - 1.0

    # Adaptive sigmoid urgency: steepness increases as global slack tightens
    slack_median = np.median(slack)
    tau_urgency = np.clip(1.0 + 0.5 * np.maximum(0.0, -slack_median), 0.5, 5.0)
    urgency_raw = 1.0 / (1.0 + np.exp(slack / tau_urgency + 1.0))  # shifted sigmoid: high score when slack < 0
    norm_urgency = normalize_rank(urgency_raw)
    urgency_term = -5.0 * norm_urgency

    # Soft-gated SEER: no hard zero — maintains differentiable gradient even for negative slack
    exec_comm_sum = min_exec_time + min_comm_time + eps
    base_seer = min_incremental_energy / exec_comm_sum
    # Smooth gating: exp(-max(0, slack)/tau) → replaced with logistic gate for continuity
    slack_gate = 1.0 / (1.0 + np.exp(slack / 3.0))  # ~1 when slack << 0, ~0.5 at slack=0, ~0 when slack >> 0
    seer_soft_gated = base_seer * slack_gate
    norm_seer = normalize_rank(seer_soft_gated)
    seer_term = -1.8 * norm_seer

    # Critical-path density: upward_rank * remaining_work / (exec+comm), penalized by uncertainty only when slack > 0
    cp_density = upward_rank * remaining_work / (exec_comm_sum + eps)
    comm_unc_penalty = np.where(slack > 0, min_comm_time * uncertainty * 0.2, 0.0)
    cp_density_penalized = cp_density + comm_unc_penalty / (exec_comm_sum + eps)
    norm_cp_density = normalize_rank(cp_density_penalized)
    cp_density_term = 1.3 * norm_cp_density

    # Fairness: linear wait boost only when slack > safe margin, scaled by relative wait rank
    safe_slack_threshold = np.clip(np.percentile(slack, 75), 0.0, np.inf)
    wait_boost = np.where(slack > safe_slack_threshold,
                         np.clip(ready_wait_time / (np.maximum(np.mean(ready_wait_time), eps) + 1.0), 0.0, 0.4),
                         0.0)
    norm_wait = normalize_rank(wait_boost)
    wait_term = -0.22 * norm_wait

    # Uncertainty boost: only for high uncertainty AND positive slack, with soft cap
    unc_boost = np.where((slack > 0.0) & (uncertainty > np.percentile(uncertainty, 80)),
                        np.clip(uncertainty * 0.2, 0.0, 0.1), 0.0)
    norm_unc = normalize_rank(unc_boost)
    unc_term = -0.06 * norm_unc

    score = urgency_term + seer_term + cp_density_term + wait_term + unc_term
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
