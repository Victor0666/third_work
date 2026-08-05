"""Configuration and scenario construction for ``drlea_nichgp``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from algorithms.llm_safe_hrl.hrl_mix.train_config import (
    CLOUD_BW_TIERS,
    CLOUD_PC_TIERS,
    EDGE_BW_TIERS,
    EDGE_PC_TIERS,
    RESOURCE_CONFIG,
    SIZE_FULL,
    TASK_DAX_FILES,
)
from project_paths import PROJECT_ROOT

from . import METHOD_ID, SCHEMA_VERSION


DDL_NAMES = {
    "T": "Tight",
    "TIGHT": "Tight",
    "M": "Medium",
    "MEDIUM": "Medium",
    "L": "Loose",
    "LOOSE": "Loose",
}
DDL_SMALL_PROBABILITY = {
    "Tight": 0.8,
    "Medium": 0.5,
    "Loose": 0.2,
}
REWARD_MODES = (
    "completion_only",
    "deadline_energy",
    "deadline_energy_slack",
)


def ensure_disjoint_seeds(
    train_seeds,
    validation_seeds,
    test_seeds,
) -> None:
    groups = [
        set(map(int, train_seeds)),
        set(map(int, validation_seeds)),
        set(map(int, test_seeds)),
    ]
    if (
        groups[0] & groups[1]
        or groups[0] & groups[2]
        or groups[1] & groups[2]
    ):
        raise ValueError(
            "train, validation and test seeds must be strictly disjoint"
        )


@dataclass(frozen=True)
class AgentConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    hidden_dims: tuple[int, int] = (128, 128)
    replay_capacity: int = 50_000
    batch_size: int = 64


@dataclass(frozen=True)
class GPConfig:
    population_size: int = 80
    generations: int = 20
    archive_size: int = 4
    tournament_size: int = 4
    elite_size: int = 4
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    max_depth: int = 6
    decision_ready_threshold: int = 6
    target_situations: int = 20
    clearing_radius: float = 0.0
    clearing_capacity: int = 1


@dataclass(frozen=True)
class RewardConfig:
    mode: str = "deadline_energy"
    alpha_deadline: float = 1.0
    beta_energy: float = 0.05
    slack_weight: float = 0.1
    energy_normalizer: float = 1_000.0
    time_normalizer: float = 300.0

    def __post_init__(self) -> None:
        if self.mode not in REWARD_MODES:
            raise ValueError(f"Unsupported reward mode: {self.mode}")


@dataclass(frozen=True)
class NormalizationConfig:
    workload_mi: float = 150_000.0
    data_bits: float = 1e10
    time_seconds: float = 300.0
    rank_seconds: float = 1_000.0
    remaining_work_mi: float = 3_000_000.0
    energy_joules: float = 1_000.0
    ready_count: float = 32.0


@dataclass(frozen=True)
class ComparisonConfig:
    method_id: str
    schema_version: int
    scenario: str
    ddl: str
    seed: int
    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    workflows_per_episode: int
    ra_episodes: int
    sa_episodes: int
    allow_busy_vm_queueing: bool
    arrival_lambda: float
    horizon: float
    fuzzy_delta1: float
    fuzzy_delta2: float
    fuzzy_deadline_eta: float
    fuzzy_energy_lambda: float
    deadline_alpha_small: float
    deadline_alpha_large: float
    deadline_alpha_small_prob: float
    dax_paths: tuple[str, ...]
    deadline_cache_path: str
    num_cloud_hosts: int
    num_edge_hosts: int
    cloud_vms_per_host: tuple[int, ...]
    edge_vms_per_host: tuple[int, ...]
    cloud_pc_tiers: tuple[float, ...]
    edge_pc_tiers: tuple[float, ...]
    cloud_bw_tiers: tuple[float, ...]
    edge_bw_tiers: tuple[float, ...]
    routing: AgentConfig = field(default_factory=AgentConfig)
    sequencing: AgentConfig = field(default_factory=AgentConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    normalization: NormalizationConfig = field(
        default_factory=NormalizationConfig
    )

    @property
    def output_dir(self) -> Path:
        short_ddl = self.ddl[0].upper()
        return (
            PROJECT_ROOT
            / "out"
            / "comparisons"
            / METHOD_ID
            / f"{self.scenario}_{short_ddl}_s{self.seed}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_scenario(value: str) -> str:
    value = str(value).strip().upper()
    if len(value) != 2 or any(code not in SIZE_FULL for code in value):
        raise ValueError("scenario must be one of SS through LL")
    return value


def normalize_ddl(value: str) -> str:
    key = str(value).strip().upper()
    if key not in DDL_NAMES:
        raise ValueError("ddl must be T/M/L or Tight/Medium/Loose")
    return DDL_NAMES[key]


def build_config(
    scenario: str = "SS",
    ddl: str = "T",
    seed: int = 0,
    *,
    workflows_per_episode: int = 50,
    ra_episodes: int = 200,
    sa_episodes: int = 200,
    reward_mode: str = "deadline_energy",
    smoke: bool = False,
) -> ComparisonConfig:
    """Build a comparison config from the current project's scenario tables."""

    scenario = normalize_scenario(scenario)
    ddl_name = normalize_ddl(ddl)
    task_code, resource_code = scenario
    task_size = SIZE_FULL[task_code]
    resource_size = SIZE_FULL[resource_code]
    resources = RESOURCE_CONFIG[resource_code]
    dax_paths = tuple(
        (Path("data") / "dax" / name).as_posix()
        for name in TASK_DAX_FILES[task_code]
    )
    missing = [
        path
        for path in dax_paths
        if not (PROJECT_ROOT / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing current-project DAX files: {missing}")
    cache_path = (
        Path("data")
        / "deadlines"
        / "fcfs"
        / (
            f"fcfs_{task_size}Task_{resource_size}Res_"
            "seed0-1000.json"
        )
    )
    if not (PROJECT_ROOT / cache_path).is_file():
        raise FileNotFoundError(
            f"Missing current-project deadline cache: {cache_path}"
        )

    if smoke:
        workflows_per_episode = 3
        ra_episodes = 2
        sa_episodes = 2
        routing = AgentConfig(
            hidden_dims=(32, 32),
            replay_capacity=512,
            batch_size=4,
            epsilon_decay=0.8,
        )
        sequencing = AgentConfig(
            hidden_dims=(32, 32),
            replay_capacity=512,
            batch_size=4,
            epsilon_decay=0.8,
        )
        gp_config = GPConfig(
            population_size=8,
            generations=2,
            elite_size=2,
            max_depth=4,
            decision_ready_threshold=2,
            target_situations=4,
        )
    else:
        routing = AgentConfig()
        sequencing = AgentConfig()
        gp_config = GPConfig()

    seed = int(seed)
    config = ComparisonConfig(
        method_id=METHOD_ID,
        schema_version=SCHEMA_VERSION,
        scenario=scenario,
        ddl=ddl_name,
        seed=seed,
        train_seeds=(seed,),
        validation_seeds=(seed + 100,),
        test_seeds=(seed + 200,),
        workflows_per_episode=int(workflows_per_episode),
        ra_episodes=int(ra_episodes),
        sa_episodes=int(sa_episodes),
        allow_busy_vm_queueing=False,
        arrival_lambda=0.03,
        horizon=1e9,
        fuzzy_delta1=0.75,
        fuzzy_delta2=1.2,
        fuzzy_deadline_eta=0.95,
        fuzzy_energy_lambda=1.0,
        deadline_alpha_small=2.0,
        deadline_alpha_large=3.0,
        deadline_alpha_small_prob=DDL_SMALL_PROBABILITY[ddl_name],
        dax_paths=dax_paths,
        deadline_cache_path=cache_path.as_posix(),
        num_cloud_hosts=int(resources[0]),
        num_edge_hosts=int(resources[1]),
        cloud_vms_per_host=tuple(resources[2]),
        edge_vms_per_host=tuple(resources[3]),
        cloud_pc_tiers=tuple(CLOUD_PC_TIERS),
        edge_pc_tiers=tuple(EDGE_PC_TIERS),
        cloud_bw_tiers=tuple(CLOUD_BW_TIERS),
        edge_bw_tiers=tuple(EDGE_BW_TIERS),
        routing=routing,
        sequencing=sequencing,
        gp=gp_config,
        reward=RewardConfig(mode=reward_mode),
    )
    ensure_disjoint_seeds(
        config.train_seeds,
        config.validation_seeds,
        config.test_seeds,
    )
    return config
