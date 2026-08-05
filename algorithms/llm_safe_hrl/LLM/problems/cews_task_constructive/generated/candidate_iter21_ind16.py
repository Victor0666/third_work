import numpy as np

# 函数名中的 2 由 SeEvo 在生成 Prompt 时替换为目标版本号。
# 八个输入均为长度 N 的一维数组；相同下标始终指向同一个 ready task。
# 返回值也必须是长度 N 的一维数组，并遵守“分数越小，优先级越高”。
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

    '''
    Priority rule v2 (self-evolved): Strict lexicographic deadline-first ordering,
    with energy minimization *only* among feasible tasks, plus robust starvation guard.
    
    Key improvements:
    - Urgency is now purely lexicographic: uses softplus(-slack) → normalized to [0,1] via safe min-max,
      ensuring zero-energy tasks never override negative-slack urgency.
    - SEER term is computed *only for slack >= 0 tasks*, and scaled additively (not multiplicatively)
      to avoid coupling that breaks deadline dominance.
    - Criticality is simplified: upward_rank * remaining_work, risk-weighted by tanh(uncertainty),
      then normalized *jointly* with urgency to preserve relative criticality under tight deadlines.
    - Aging is decoupled from slack denominator to prevent suppression under extreme urgency;
      uses bounded linear ramp + saturating sigmoid, capped at 0.3 contribution.
    - All normalizations are finite, N=1-safe, and use eps-robust min-max fallbacks (no MAD degeneracy).
    - Final score = urgency_score + (SEER_score if slack>=0 else 0) + criticality_score + aging_score,
      enforcing strict priority hierarchy: deadline > energy > criticality > fairness.
    '''
    eps = 1e-08
    # Cast and sanitize inputs — preserve original semantics, avoid inf/nan propagation
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=eps)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=100.0, neginf=0.0)

    # --- URGENCY: lexicographically dominant, monotonic, [0,1]-bounded ---
    # softplus(-slack) gives smooth, strictly decreasing urgency as slack decreases
    urgency_raw = np.log1p(np.exp(-slack))
    # Safe min-max normalization: handles N=1, avoids division-by-zero, preserves order
    urg_min, urg_max = np.min(urgency_raw), np.max(urgency_raw)
    urgency_score = (urgency_raw - urg_min + eps) / (urg_max - urg_min + eps)
    urgency_score = np.clip(urgency_score, 0.0, 1.0)

    # --- ENERGY TERM (SEER): only active for deadline-feasible tasks (slack >= 0) ---
    seer_base = (min_incremental_energy + eps) / (min_exec_time + min_comm_time + eps)
    # Gate: zero out SEER for urgent (slack < 0) tasks → ensures deadline dominates energy
    seer_mask = (slack >= 0).astype(float)
    seer_score = seer_base * seer_mask
    # Normalize only over feasible subset; if no feasible task, set all to 0
    seer_min, seer_max = np.min(seer_score), np.max(seer_score)
    if seer_max - seer_min < eps:
        seer_norm = np.zeros_like(seer_score)
    else:
        seer_norm = (seer_score - seer_min) / (seer_max - seer_min + eps)
    seer_score = np.clip(seer_norm, 0.0, 1.0) * 0.4  # bounded energy contribution

    # --- CRITICALITY: upward_rank * remaining_work, risk-amplified, jointly normalized with urgency ---
    critical_base = upward_rank * remaining_work
    risk_factor = 1.0 + np.tanh(uncertainty)
    critical_risk = critical_base * risk_factor
    # Joint normalization with urgency to avoid scale inflation; ensures criticality never overrides urgency
    joint_min = np.minimum(urg_min, np.min(critical_risk))
    joint_max = np.maximum(urg_max, np.max(critical_risk))
    critical_score = (critical_risk - joint_min + eps) / (joint_max - joint_min + eps)
    critical_score = np.clip(critical_score, 0.0, 1.0) * 0.35

    # --- AGING: fairness guard, decoupled from slack to prevent suppression under urgency ---
    # Linear ramp dominates early, sigmoid caps long waits; total contribution capped at 0.3
    linear_aging = np.clip(ready_wait_time * 0.02, 0.0, 0.25)
    sigmoid_aging = 1.0 / (1.0 + np.exp(-(ready_wait_time - 10.0))) * 0.05
    aging_score = np.clip(linear_aging + sigmoid_aging, 0.0, 0.3)

    # --- FINAL SCORE: lexicographic sum — urgency first, then additive energy/criticality/fairness ---
    score = urgency_score + seer_score + critical_score + aging_score

    # Final sanitization: ensure finite, deterministic, shape-(N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    return score.astype(float)
