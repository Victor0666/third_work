import numpy as np
def get_task_priority_v2(
    min_exec_time, min_comm_time, min_incremental_energy, slack,
    upward_rank, remaining_work, ready_wait_time, uncertainty
):
    return np.asarray(slack, dtype=float)
