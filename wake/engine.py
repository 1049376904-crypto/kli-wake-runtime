"""Kli Wake Runtime 核心：连续状态演化 + hazard 累积。

这个模块是纯的：不碰数据库、不碰 HTTP、不认识 Agent、不发消息。
它只回答一个问题——"到这一分钟，是否产生了一次 spontaneous WakeOpportunity"。

钉死的语义（对应方案文档的架构不变量）：

1. 时间网格固定。所有评估时刻是 UTC epoch 整分钟，单步 dt 恒为 policy.step_seconds。
   不允许用"实际经过了多少秒"当 dt——否则连续运行和重启续跑的步长边界不同，
   轨迹必然分叉，第 9 条不变量（确定性重放）就是假的。

2. RNG 完整状态随 ActivationState 持久化（不是只存 seed）。每步消耗顺序固定：
   先 X，再 T，D 不消耗；仅在换 cycle 时追加一次 theta 抽取。

3. 长间隔恢复拆成三件独立的事（advance 的 gap 分支）：
   - 状态照常按整分钟步进演化（D 该回归就回归，T/X 该漂就漂，RNG 照常消耗）
   - 这段时间的 hazard 增量丢弃
   - 当前 cycle 作废，Θ 重抽，H 归零，记一条 suppressed_spontaneous
   自发 Wake 是机会，不是欠账。绝不补发。

4. Direct Wake（Calendar / MCP / iOS 等精确事件）不属于这个模块。它只经由
   report_agent_run() 影响 D，禁止触碰 cycle 的 hazard 和 theta。见该函数注释。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any, Iterable

from .policy import DEFAULT_POLICY, WakePolicy

SECONDS_PER_MINUTE = 60


# --------------------------------------------------------------------------
# 时间网格
# --------------------------------------------------------------------------

def epoch_minute(unix_seconds: float) -> int:
    """把 wall-clock 秒量化到 UTC epoch 整分钟（向下取整）。

    残余的不足一分钟不做插值，留给下一次 tick。
    """
    return int(unix_seconds // SECONDS_PER_MINUTE)


def minute_to_epoch_seconds(minute: int) -> int:
    return minute * SECONDS_PER_MINUTE


# --------------------------------------------------------------------------
# 事件
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class WakeEvent:
    """引擎产出的事件。engine 不投递任何东西，只把事件交回调用方。"""

    kind: str  # spontaneous_wake / suppressed_spontaneous / clock_regression / catchup_truncated
    minute: int
    cycle_id: str
    detail: dict[str, Any]


# --------------------------------------------------------------------------
# RNG：完整状态可序列化
# --------------------------------------------------------------------------

def _rng(state_tuple) -> random.Random:
    rng = random.Random()
    rng.setstate(_rng_state_from_json(state_tuple))
    return rng


def _rng_state_to_json(st) -> list:
    version, keys, gauss_next = st
    return [int(version), [int(k) for k in keys], gauss_next]


def _rng_state_from_json(v):
    version, keys, gauss_next = v
    return (
        int(version),
        tuple(int(k) for k in keys),
        None if gauss_next is None else float(gauss_next),
    )


def _normal(rng: random.Random) -> float:
    """标准正态。

    用 normalvariate 而不是 gauss：gauss 会缓存一个备用值（gauss_next），
    让"消耗了几个随机数"和状态耦合得更微妙。normalvariate 无缓存，行为更好推理。
    """
    return rng.normalvariate(0.0, 1.0)


def _draw_theta(rng: random.Random) -> float:
    """Θ ~ Exp(1)，每个 cycle 只抽一次。

    用 1 - random() 把区间挪到 (0, 1]，否则 random() 恰好返回 0.0 时 Θ = +inf。
    """
    u = 1.0 - rng.random()
    return -math.log(u)


# --------------------------------------------------------------------------
# 状态
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ActivationState:
    """权威激活状态。持续保存、持续演化，只在第一次没有状态时才初始化。"""

    minute: int  # lastEvaluatedAt，UTC epoch 整分钟
    drive: float  # D
    tone: float  # T
    drift: float  # X
    hazard: float  # H，当前 cycle 内累积
    theta: float  # Θ，当前 cycle 内固定
    cycle_id: str
    cycle_started_minute: int
    cycle_seq: int
    rng_state: list
    state_version: int
    policy_version: str
    warmup_started_minute: int
    warmup_complete: bool
    runs_total: int = 0  # 真实 Agent Run 累计次数（含 UserTurn / Direct / Spontaneous）

    # ---- 序列化：可直接进 SQLite 的 JSON 列 ----

    def to_json(self) -> dict:
        return {
            "minute": self.minute,
            "drive": self.drive,
            "tone": self.tone,
            "drift": self.drift,
            "hazard": self.hazard,
            "theta": self.theta,
            "cycle_id": self.cycle_id,
            "cycle_started_minute": self.cycle_started_minute,
            "cycle_seq": self.cycle_seq,
            "rng_state": _rng_state_to_json(_rng_state_from_json(self.rng_state)),
            "state_version": self.state_version,
            "policy_version": self.policy_version,
            "warmup_started_minute": self.warmup_started_minute,
            "warmup_complete": self.warmup_complete,
            "runs_total": self.runs_total,
        }

    @staticmethod
    def from_json(d: dict) -> "ActivationState":
        return ActivationState(
            minute=int(d["minute"]),
            drive=float(d["drive"]),
            tone=float(d["tone"]),
            drift=float(d["drift"]),
            hazard=float(d["hazard"]),
            theta=float(d["theta"]),
            cycle_id=str(d["cycle_id"]),
            cycle_started_minute=int(d["cycle_started_minute"]),
            cycle_seq=int(d["cycle_seq"]),
            rng_state=_rng_state_to_json(_rng_state_from_json(d["rng_state"])),
            state_version=int(d["state_version"]),
            policy_version=str(d["policy_version"]),
            warmup_started_minute=int(d["warmup_started_minute"]),
            warmup_complete=bool(d["warmup_complete"]),
            runs_total=int(d.get("runs_total", 0)),
        )


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _cycle_id(minute: int, seq: int) -> str:
    return f"cyc_{minute:09d}_{seq:06d}"


def init_state(
    now_minute: int,
    policy: WakePolicy = DEFAULT_POLICY,
    seed: int | None = None,
    cold_start: bool = True,
) -> ActivationState:
    """只在系统第一次完全没有 ActivationState 时调用。

    服务重启必须读旧状态，绝不重新初始化——否则每次部署都会重抽节律。
    """
    rng = random.Random(seed)
    if cold_start:
        drive = _clamp(policy.warmup_d_initial, policy.d_min, policy.d_max)
        tone = _clamp(policy.warmup_t_initial, policy.t_min, policy.t_max)
    else:
        drive, tone = policy.d_mean, policy.t_mean
    theta = _draw_theta(rng)
    return ActivationState(
        minute=now_minute,
        drive=drive,
        tone=tone,
        drift=policy.x_mean,
        hazard=0.0,
        theta=theta,
        cycle_id=_cycle_id(now_minute, 0),
        cycle_started_minute=now_minute,
        cycle_seq=0,
        rng_state=_rng_state_to_json(rng.getstate()),
        state_version=1,
        policy_version=policy.version,
        warmup_started_minute=now_minute,
        warmup_complete=not cold_start,
    )


# --------------------------------------------------------------------------
# λ(t)
# --------------------------------------------------------------------------

def lambda_per_hour(
    drive: float,
    tone: float,
    drift: float,
    policy: WakePolicy = DEFAULT_POLICY,
    mmod: float = 1.0,
) -> float:
    """瞬时唤醒倾向，单位 次/小时。

    λ 不是概率，也不是"发消息概率"。它只决定获得一次运行机会的瞬时倾向。
    """
    m = _clamp(mmod, policy.mmod_min, policy.mmod_max)
    z = (
        policy.beta_d * (drive - policy.d_mean)
        + policy.beta_t * (tone - policy.t_mean)
        + policy.beta_x * (drift - policy.x_mean)
    )
    lam = policy.lambda_base_per_hour * math.exp(z) * m
    return _clamp(lam, policy.lambda_min_per_hour, policy.lambda_max_per_hour)


def lambda_of_state(
    state: ActivationState, policy: WakePolicy = DEFAULT_POLICY, mmod: float = 1.0
) -> float:
    return lambda_per_hour(state.drive, state.tone, state.drift, policy, mmod)


# --------------------------------------------------------------------------
# 状态演化（单步）
# --------------------------------------------------------------------------

def _rho(dt_minutes: float, tau_minutes: float) -> float:
    return 2.0 ** (-dt_minutes / tau_minutes)


def _evolve_drive(d: float, dt: float, p: WakePolicy) -> float:
    """D：纯均值回归，无噪声。时间本身不会把 D 推得越来越高。"""
    rho = _rho(dt, p.d_tau_minutes)
    return _clamp(p.d_mean + (d - p.d_mean) * rho, p.d_min, p.d_max)


def _evolve_tone(t: float, dt: float, p: WakePolicy, rng: random.Random) -> float:
    """T：慢 OU 过程。某个下午可能整体偏活跃，但不会一分钟一变。"""
    rho = _rho(dt, p.t_tau_minutes)
    nxt = (
        p.t_mean
        + (t - p.t_mean) * rho
        + p.t_sigma * math.sqrt(1.0 - rho * rho) * _normal(rng)
    )
    return _clamp(nxt, p.t_min, p.t_max)


def _evolve_drift(x: float, dt: float, p: WakePolicy, rng: random.Random) -> float:
    """X：有惯性的短期随机漂移，负责"这一小阵突然更活跃/更安静"。"""
    rho = _rho(dt, p.x_tau_minutes)
    nxt = x * rho + p.x_sigma * math.sqrt(1.0 - rho * rho) * _normal(rng)
    return _clamp(nxt, p.x_min, p.x_max)


def step(
    state: ActivationState,
    policy: WakePolicy = DEFAULT_POLICY,
    mmod: float = 1.0,
) -> tuple[ActivationState, bool]:
    """推进一个网格步（默认 60s）。返回 (新状态, 本步是否产生 spontaneous wake)。

    RNG 消耗顺序：X → T →（若 wake）theta。改动这个顺序会让所有历史轨迹不可重放。

    hazard 用右端点矩形法累积：H += λ(演化后状态) · dt。
    在 60s 步长下相对分钟级的状态时间尺度足够精确，而且比梯形法少一次状态耦合。
    H = ∫λ 没有闭式解（λ 依赖含噪声的 D/T/X），只能数值积分。
    """
    rng = _rng(state.rng_state)
    dt = policy.step_minutes

    drift = _evolve_drift(state.drift, dt, policy, rng)
    tone = _evolve_tone(state.tone, dt, policy, rng)
    drive = _evolve_drive(state.drive, dt, policy)

    lam = lambda_per_hour(drive, tone, drift, policy, mmod)
    hazard = state.hazard + lam * policy.step_hours
    minute = state.minute + 1

    wake = hazard >= state.theta
    theta = state.theta
    cycle_id = state.cycle_id
    cycle_started = state.cycle_started_minute
    cycle_seq = state.cycle_seq

    if wake:
        # 产生机会后立刻开新 cycle：Θ 重抽一次，H 归零。
        theta = _draw_theta(rng)
        hazard = 0.0
        cycle_seq += 1
        cycle_id = _cycle_id(minute, cycle_seq)
        cycle_started = minute

    warmup_complete = state.warmup_complete or (
        minute - state.warmup_started_minute >= policy.warmup_minutes
    )

    nxt = replace(
        state,
        minute=minute,
        drive=drive,
        tone=tone,
        drift=drift,
        hazard=hazard,
        theta=theta,
        cycle_id=cycle_id,
        cycle_started_minute=cycle_started,
        cycle_seq=cycle_seq,
        rng_state=_rng_state_to_json(rng.getstate()),
        state_version=state.state_version + 1,
        warmup_complete=warmup_complete,
    )
    return nxt, wake


def _evolve_only(
    state: ActivationState, steps: int, policy: WakePolicy
) -> ActivationState:
    """只演化 D/T/X 和 RNG，不累积 hazard、不判定 wake。用于长间隔恢复。"""
    rng = _rng(state.rng_state)
    dt = policy.step_minutes
    drive, tone, drift = state.drive, state.tone, state.drift
    for _ in range(steps):
        drift = _evolve_drift(drift, dt, policy, rng)
        tone = _evolve_tone(tone, dt, policy, rng)
        drive = _evolve_drive(drive, dt, policy)
    return replace(
        state,
        drive=drive,
        tone=tone,
        drift=drift,
        rng_state=_rng_state_to_json(rng.getstate()),
    )


def _restart_cycle(state: ActivationState, minute: int) -> ActivationState:
    rng = _rng(state.rng_state)
    theta = _draw_theta(rng)
    seq = state.cycle_seq + 1
    return replace(
        state,
        hazard=0.0,
        theta=theta,
        cycle_id=_cycle_id(minute, seq),
        cycle_started_minute=minute,
        cycle_seq=seq,
        rng_state=_rng_state_to_json(rng.getstate()),
    )


# --------------------------------------------------------------------------
# 推进到目标时刻（含恢复语义）
# --------------------------------------------------------------------------

def advance(
    state: ActivationState,
    target_minute: int,
    policy: WakePolicy = DEFAULT_POLICY,
    mmod: float = 1.0,
) -> tuple[ActivationState, list[WakeEvent]]:
    """把状态推进到 target_minute，返回 (新状态, 事件列表)。

    三种情况：

    - target <= state.minute：时钟回拨或重复 tick。忽略，记 anomaly。
      dt 应该由 monotonic clock 算，wall clock 只用来记时间戳。
    - 空档 ≤ policy.max_gap_minutes：逐分钟正常演化并累积 hazard，可能产出多个机会。
    - 空档 > policy.max_gap_minutes：状态照常演化到 target，但 hazard 增量丢弃、
      cycle 作废重抽，只记一条 suppressed_spontaneous。停机期间错过的不补发。

    注意 mmod 在整段推进里按常量处理。调用方若要更精细，应该分段调用 advance。
    """
    events: list[WakeEvent] = []
    gap = target_minute - state.minute

    if gap <= 0:
        if gap < 0:
            events.append(
                WakeEvent(
                    kind="clock_regression",
                    minute=state.minute,
                    cycle_id=state.cycle_id,
                    detail={"target_minute": target_minute, "gap_minutes": gap},
                )
            )
        return state, events

    if gap > policy.max_gap_minutes:
        steps = min(gap, policy.max_catchup_minutes)
        if steps < gap:
            events.append(
                WakeEvent(
                    kind="catchup_truncated",
                    minute=target_minute,
                    cycle_id=state.cycle_id,
                    detail={"gap_minutes": gap, "replayed_minutes": steps},
                )
            )
        nxt = _evolve_only(state, steps, policy)
        nxt = _restart_cycle(nxt, target_minute)
        nxt = replace(
            nxt,
            minute=target_minute,
            state_version=state.state_version + 1,
            warmup_complete=state.warmup_complete
            or (target_minute - state.warmup_started_minute >= policy.warmup_minutes),
        )
        events.append(
            WakeEvent(
                kind="suppressed_spontaneous",
                minute=target_minute,
                cycle_id=nxt.cycle_id,
                detail={
                    "gap_minutes": gap,
                    "discarded_hazard": state.hazard,
                    "theta_of_discarded_cycle": state.theta,
                    "reason": "gap_exceeds_max_gap_minutes",
                },
            )
        )
        return nxt, events

    cur = state
    for _ in range(gap):
        cur, wake = step(cur, policy, mmod)
        if wake:
            events.append(
                WakeEvent(
                    kind="spontaneous_wake",
                    minute=cur.minute,
                    cycle_id=cur.cycle_id,  # 已是新 cycle；旧 cycle 在 detail 里
                    detail={
                        "lambda_per_hour": lambda_of_state(cur, policy, mmod),
                        "drive": cur.drive,
                        "tone": cur.tone,
                        "drift": cur.drift,
                        "mmod": _clamp(mmod, policy.mmod_min, policy.mmod_max),
                    },
                )
            )
    return cur, events


# --------------------------------------------------------------------------
# Agent Run 反馈
# --------------------------------------------------------------------------

def report_agent_run(
    state: ActivationState, policy: WakePolicy = DEFAULT_POLICY
) -> ActivationState:
    """任何真实 Agent Run 结束后调用：D ← clamp(D - k_run)。

    "任何"包括 UserTurn、Direct Wake（Calendar / MCP / iOS）和 Spontaneous Wake。
    应该挂在 AgentRuntime 的统一出口，不要让每个调用方自己记得触发。

    这不是 cooldown，只是刚跑完之后短期稍微安静一点。

    关键约束：这里绝不触碰 hazard、theta、cycle_id。
    Direct Wake 只额外注入一个机会 + kick D，不干扰自发节律。
    如果 Direct Wake 很频繁导致 D 持续偏低、自发唤醒变少——这是预期行为，不是 bug。
    也不消耗 RNG，所以不会让轨迹分叉。
    """
    drive = _clamp(state.drive - policy.k_run, policy.d_min, policy.d_max)
    return replace(
        state,
        drive=drive,
        state_version=state.state_version + 1,
        runs_total=state.runs_total + 1,
    )


# --------------------------------------------------------------------------
# 只读诊断（只写日志，绝不进 Agent 上下文）
# --------------------------------------------------------------------------

def diagnostics(
    state: ActivationState, policy: WakePolicy = DEFAULT_POLICY, mmod: float = 1.0
) -> dict:
    """给 metrics / 调参曲线用。

    这里的每一个字段都不允许出现在 Agent 的 prompt 里：模型看到
    "系统这么高概率叫醒我" 就会反向推断"所以我应该很想她"。
    """
    lam = lambda_of_state(state, policy, mmod)
    return {
        "minute": state.minute,
        "drive": round(state.drive, 4),
        "tone": round(state.tone, 4),
        "drift": round(state.drift, 4),
        "lambda_per_hour": round(lam, 4),
        "hazard": round(state.hazard, 4),
        "theta": round(state.theta, 4),
        "hazard_remaining": round(max(0.0, state.theta - state.hazard), 4),
        "cycle_id": state.cycle_id,
        "cycle_age_minutes": state.minute - state.cycle_started_minute,
        "state_version": state.state_version,
        "policy_version": state.policy_version,
        "warmup_complete": state.warmup_complete,
        "runs_total": state.runs_total,
    }


def replay(
    state: ActivationState,
    minutes: int,
    policy: WakePolicy = DEFAULT_POLICY,
    mmod: float = 1.0,
) -> Iterable[tuple[ActivationState, bool]]:
    """逐步生成器，供模拟器和重放测试使用。生产与模拟共享同一 step()。"""
    cur = state
    for _ in range(minutes):
        cur, wake = step(cur, policy, mmod)
        yield cur, wake
