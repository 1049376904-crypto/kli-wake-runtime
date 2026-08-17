"""WakeActivationPolicy —— 冻结的策略参数对象。

所有可调参数只存在于这里，不散落在引擎代码里。每个 ActivationCycle 记录
policy_version，这样以后改成 wake-v1.1，旧 cycle 仍然知道自己是按哪套规则跑的。

参数默认值来自《让TA自己醒来 V1 基线》第 12 节，是开发校准起点，不是心理学常量。
调参前先读 README 的"太吵/太安静"一节，不要一次动多个旋钮。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

POLICY_VERSION = "wake-v1.0-dev"


@dataclass(frozen=True)
class WakePolicy:
    """不可变策略对象。用 with_overrides() 派生变体，不要原地改字段。"""

    version: str = POLICY_VERSION

    # ---- 时间网格 ----------------------------------------------------------
    # 单步长度。所有评估时刻量化到 UTC epoch 整分钟，dt 恒为这个值。
    # 固定网格是确定性重放的前提：不允许用"实际经过了多少秒"当 dt。
    step_seconds: int = 60

    # ---- activationDrive (D) 短期激活驱动力，无噪声 ------------------------
    d_mean: float = 0.50
    d_min: float = 0.20
    d_max: float = 0.80
    d_tau_minutes: float = 12.0
    k_run: float = 0.10  # 每次真实 Agent Run 后的负向 kick（不是 cooldown）

    # ---- latentActivityTone (T) 几小时尺度的活跃底色 -----------------------
    t_mean: float = 0.50
    t_min: float = 0.25
    t_max: float = 0.75
    t_tau_minutes: float = 360.0
    t_sigma: float = 0.10

    # ---- stochasticDriftState (X) 有惯性的短期随机漂移 ---------------------
    x_mean: float = 0.00
    x_min: float = -0.40
    x_max: float = 0.40
    x_tau_minutes: float = 25.0
    x_sigma: float = 0.18

    # ---- λ(t) --------------------------------------------------------------
    lambda_base_per_hour: float = 1.50  # 整体密度总旋钮，优先调它
    beta_d: float = 1.8
    beta_t: float = 1.6
    beta_x: float = 1.2
    lambda_min_per_hour: float = 0.15  # 数值护栏，不是调参旋钮
    lambda_max_per_hour: float = 8.00  # 数值护栏，限制极端 burst

    # ---- Modulation 总边界 -------------------------------------------------
    # 上游（EB / 记忆 / OpenLoop）的聚合结果在进入引擎前后都被夹在这个区间。
    # 聚合本身在 modulation 层做：ln(Mmod) = clamp(Σ conf_i·ln(m_i), ln lo, ln hi)
    mmod_min: float = 0.60
    mmod_max: float = 3.00

    # ---- 冷启动 warm-up ----------------------------------------------------
    # 首次部署时 D/T 略低于中性，避免"刚上线就频繁主动"。
    # 状态会自然向 μ 回归，不需要额外的时变系数。
    warmup_d_initial: float = 0.35
    warmup_t_initial: float = 0.45
    warmup_minutes: int = 480  # 8h

    # ---- 恢复语义 ----------------------------------------------------------
    # 空档 ≤ max_gap_minutes：正常逐分钟演化并累积 hazard。
    # 空档 > max_gap_minutes：状态照常演化，hazard 增量丢弃，cycle 作废重抽。
    max_gap_minutes: int = 15
    # 极长停机时最多回放这么多分钟的状态演化（14 天）。
    # 状态最慢的时间尺度是 T 的 6h，14 天远超混合时间，更早的演化已无信息量。
    max_catchup_minutes: int = 20160

    def __post_init__(self) -> None:
        if self.step_seconds <= 0:
            raise ValueError("step_seconds 必须为正")
        if not (self.d_min <= self.d_mean <= self.d_max):
            raise ValueError("d_mean 必须落在 [d_min, d_max] 内")
        if not (self.t_min <= self.t_mean <= self.t_max):
            raise ValueError("t_mean 必须落在 [t_min, t_max] 内")
        if not (self.x_min <= self.x_mean <= self.x_max):
            raise ValueError("x_mean 必须落在 [x_min, x_max] 内")
        if min(self.d_tau_minutes, self.t_tau_minutes, self.x_tau_minutes) <= 0:
            raise ValueError("所有 tau 必须为正")
        if not (0 < self.lambda_min_per_hour <= self.lambda_max_per_hour):
            raise ValueError("需要 0 < lambda_min <= lambda_max")
        if not (0 < self.mmod_min <= 1.0 <= self.mmod_max):
            raise ValueError("Mmod clamp 区间必须包含 1.0")
        if self.max_gap_minutes < 1:
            raise ValueError("max_gap_minutes 至少为 1")

    # ---- 派生与指纹 --------------------------------------------------------

    def with_overrides(self, **kwargs) -> "WakePolicy":
        """派生一个新策略。若改动了非 version 字段，version 自动加上指纹后缀，
        避免模拟结果被误标成基线 policy_version。"""
        substantive = {k: v for k, v in kwargs.items() if k != "version"}
        derived = replace(self, **kwargs)
        if substantive and "version" not in kwargs:
            derived = replace(derived, version=f"{self.version}+{derived.fingerprint()}")
        return derived

    def fingerprint(self) -> str:
        """参数内容的短哈希，用于区分"同一版本号但参数被改过"的情况。"""
        payload = {k: v for k, v in asdict(self).items() if k != "version"}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def step_minutes(self) -> float:
        return self.step_seconds / 60.0

    @property
    def step_hours(self) -> float:
        return self.step_seconds / 3600.0


DEFAULT_POLICY = WakePolicy()
