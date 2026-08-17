#!/usr/bin/env python3
"""离线模拟器：跑几十天曲线，看这套参数到底吵不吵。

和生产共享同一个 engine.step()，不复制一份公式——否则调参结论对不上生产行为。

用法：

    python -m wake.simulate --days 30 --seed 7
    python -m wake.simulate --days 30 --user-turns-per-day 12 --silent-rate 0.7
    python -m wake.simulate --days 30 --csv /tmp/wake.csv
    python -m wake.simulate --days 14 --sweep lambda_base_per_hour=1.2,1.5,1.8
    python -m wake.simulate --days 14 --sweep k_run=0.06,0.10,0.14

注意输出里的两组数字含义完全不同：
  - wake 密度 = 引擎给了多少次"运行机会"
  - 感知密度 = 用户实际会看到几条消息（叠加假设 silent rate 和表达预算硬闸）
第二组才是体验。第一组高不一定吵。
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from collections import deque
from dataclasses import dataclass

from .engine import (
    diagnostics,
    init_state,
    lambda_of_state,
    report_agent_run,
    step,
)
from .policy import DEFAULT_POLICY, WakePolicy

# 表达预算（硬闸）默认值。
# 这是唯一不依赖模型自律的兜底：evidence / value 都是模型自己填的布尔，
# confidence 又明确不设阈值，所以 silent 完全押在 prompt 纪律上。
# 加了这道闸，最坏情况从"连说二十句"变成"说满额度就闭嘴"。
# 它只作用于 visibility=external，与 λ 完全解耦，不扭曲随机质感。
EXTERNAL_PER_HOUR = 2
EXTERNAL_PER_DAY = 8
EXTERNAL_PER_DAY_WARMUP = 3


@dataclass
class SimResult:
    policy: WakePolicy
    minutes: int
    wake_minutes: list[int]
    user_turn_minutes: list[int]
    expressed_minutes: list[int]
    budget_suppressed: int
    lambda_samples: list[float]
    drive_samples: list[float]
    tone_samples: list[float]

    @property
    def days(self) -> float:
        return self.minutes / 1440.0


class ExpressionBudget:
    """滚动窗口配额。超限时把 outcome 强制改写为 silent，并记 budget_suppressed。"""

    def __init__(self, per_hour: int, per_day: int) -> None:
        self.per_hour = per_hour
        self.per_day = per_day
        self._recent: deque[int] = deque()

    def allow(self, minute: int) -> bool:
        while self._recent and minute - self._recent[0] >= 1440:
            self._recent.popleft()
        in_hour = sum(1 for m in self._recent if minute - m < 60)
        if in_hour >= self.per_hour:
            return False
        if len(self._recent) >= self.per_day:
            return False
        self._recent.append(minute)
        return True


def _wants_external(scenario: random.Random, silent_rate: float) -> bool:
    return scenario.random() >= silent_rate


def simulate(
    days: float,
    policy: WakePolicy = DEFAULT_POLICY,
    seed: int = 1,
    user_turns_per_day: float = 0.0,
    silent_rate: float = 0.7,
    mmod: float = 1.0,
    cold_start: bool = True,
    csv_path: str | None = None,
) -> SimResult:
    minutes = int(days * 1440)
    state = init_state(0, policy, seed=seed, cold_start=cold_start)

    # 场景随机源（用户何时说话、Agent 是否选择表达）与轨迹 RNG 分开，
    # 这样改 silent_rate 不会改变 D/T/X 的轨迹，参数对比才有意义。
    scenario = random.Random(seed * 2654435761 % (2**31))

    budget_warm = ExpressionBudget(EXTERNAL_PER_HOUR, EXTERNAL_PER_DAY_WARMUP)
    budget_full = ExpressionBudget(EXTERNAL_PER_HOUR, EXTERNAL_PER_DAY)

    wake_minutes: list[int] = []
    user_turn_minutes: list[int] = []
    expressed_minutes: list[int] = []
    budget_suppressed = 0
    lam_s: list[float] = []
    d_s: list[float] = []
    t_s: list[float] = []

    writer = None
    fh = None
    if csv_path:
        fh = open(csv_path, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(
            [
                "minute",
                "drive",
                "tone",
                "drift",
                "lambda_per_hour",
                "hazard",
                "theta",
                "wake",
                "user_turn",
                "expressed",
            ]
        )

    p_turn = user_turns_per_day / 1440.0

    try:
        for _ in range(minutes):
            state, wake = step(state, policy, mmod)
            m = state.minute

            lam_s.append(lambda_of_state(state, policy, mmod))
            d_s.append(state.drive)
            t_s.append(state.tone)

            user_turn = p_turn > 0 and scenario.random() < p_turn
            expressed = False

            if wake:
                wake_minutes.append(m)
                # Agency Gate 的粗糙代理：以 1-silent_rate 的概率想表达。
                # 真实系统里这是模型的结构化决策，不是抛硬币；这里只为估感知密度。
                if _wants_external(scenario, silent_rate):
                    b = budget_full if state.warmup_complete else budget_warm
                    if b.allow(m):
                        expressed = True
                        expressed_minutes.append(m)
                    else:
                        budget_suppressed += 1
                # 自发唤醒真的让 Agent 跑过一次 → kick D
                state = report_agent_run(state, policy)

            if user_turn:
                user_turn_minutes.append(m)
                # UserTurn 也是真实 Agent Run，同样 kick D。很容易漏掉这一条。
                state = report_agent_run(state, policy)

            if writer is not None:
                dg = diagnostics(state, policy, mmod)
                writer.writerow(
                    [
                        m,
                        dg["drive"],
                        dg["tone"],
                        dg["drift"],
                        dg["lambda_per_hour"],
                        dg["hazard"],
                        dg["theta"],
                        int(wake),
                        int(user_turn),
                        int(expressed),
                    ]
                )
    finally:
        if fh is not None:
            fh.close()

    return SimResult(
        policy=policy,
        minutes=minutes,
        wake_minutes=wake_minutes,
        user_turn_minutes=user_turn_minutes,
        expressed_minutes=expressed_minutes,
        budget_suppressed=budget_suppressed,
        lambda_samples=lam_s,
        drive_samples=d_s,
        tone_samples=t_s,
    )


def _intervals(minutes: list[int]) -> list[int]:
    return [b - a for a, b in zip(minutes, minutes[1:])]


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return float(s[idx])


def _max_per_window(minutes: list[int], window: int) -> int:
    best = 0
    dq: deque[int] = deque()
    for m in minutes:
        dq.append(m)
        while dq and m - dq[0] >= window:
            dq.popleft()
        best = max(best, len(dq))
    return best


def report(r: SimResult) -> None:
    p = r.policy
    iv = _intervals(r.wake_minutes)
    ev = _intervals(r.expressed_minutes)

    print(f"policy            {p.version}  (fingerprint {p.fingerprint()})")
    print(f"span              {r.days:.1f} 天 / {r.minutes} 分钟")
    print(
        f"λθ={p.lambda_base_per_hour}/h  k_run={p.k_run}  τD={p.d_tau_minutes}min  "
        f"σX={p.x_sigma}  σT={p.t_sigma}  λmax={p.lambda_max_per_hour}/h"
    )
    print()
    print("— 运行机会（引擎产出）")
    print(
        f"  spontaneous wake  {len(r.wake_minutes)} 次，"
        f"{len(r.wake_minutes) / r.days:.1f} 次/天"
    )
    if iv:
        print(
            f"  间隔(min)         p10={_pct(iv, 0.1):.0f}  中位={_pct(iv, 0.5):.0f}  "
            f"p90={_pct(iv, 0.9):.0f}  最长沉默={max(iv)}"
        )
        print(
            f"  burst             间隔<5min 占比 {sum(1 for x in iv if x < 5) / len(iv):.1%}"
            f"，单小时最多 {_max_per_window(r.wake_minutes, 60)} 次"
        )
    print(
        f"  λ(t)              中位 {statistics.median(r.lambda_samples):.2f}/h  "
        f"p95 {_pct(r.lambda_samples, 0.95):.2f}/h  峰值 {max(r.lambda_samples):.2f}/h"
    )
    print(
        f"  D                 中位 {statistics.median(r.drive_samples):.3f}   "
        f"T 区间 [{min(r.tone_samples):.3f}, {max(r.tone_samples):.3f}]"
    )
    if r.user_turn_minutes:
        print(f"  UserTurn          {len(r.user_turn_minutes)} 次（也 kick D）")
    print()
    print("— 用户感知（假设 silent rate + 表达预算硬闸）")
    print(
        f"  external 表达     {len(r.expressed_minutes)} 次，"
        f"{len(r.expressed_minutes) / r.days:.1f} 次/天"
    )
    if ev:
        print(
            f"  表达间隔(min)     中位={_pct(ev, 0.5):.0f}  p90={_pct(ev, 0.9):.0f}  "
            f"最长安静={max(ev)}（≈{max(ev) / 60:.1f}h）"
        )
        print(
            f"  单小时最多        {_max_per_window(r.expressed_minutes, 60)} 条"
            f"（闸门 {EXTERNAL_PER_HOUR}）；单日最多 "
            f"{_max_per_window(r.expressed_minutes, 1440)} 条（闸门 {EXTERNAL_PER_DAY}）"
        )
    print(f"  被预算拦下        {r.budget_suppressed} 次 → 强制 silent_budget_exhausted")


def _coerce(name: str, raw: str):
    current = getattr(DEFAULT_POLICY, name)
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Kli Wake Runtime 离线模拟器")
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mmod", type=float, default=1.0, help="恒定 Mmod，测调制影响用")
    ap.add_argument("--user-turns-per-day", type=float, default=0.0)
    ap.add_argument(
        "--silent-rate",
        type=float,
        default=0.7,
        help="假设 Agency Gate 有多大比例选择 silent（只影响感知密度估计）",
    )
    ap.add_argument("--no-cold-start", action="store_true", help="跳过 warm-up，直接中性起步")
    ap.add_argument("--csv", dest="csv_path", default=None)
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖策略参数，如 --set lambda_base_per_hour=1.2",
    )
    ap.add_argument(
        "--sweep",
        default=None,
        metavar="KEY=V1,V2,...",
        help="扫一个参数的多个取值，逐个跑并对比",
    )
    args = ap.parse_args(argv)

    policy = DEFAULT_POLICY
    overrides = {}
    for item in args.set:
        if "=" not in item:
            ap.error(f"--set 需要 KEY=VALUE，收到 {item!r}")
        k, v = item.split("=", 1)
        if not hasattr(policy, k):
            ap.error(f"未知策略参数 {k!r}")
        overrides[k] = _coerce(k, v)
    if overrides:
        policy = policy.with_overrides(**overrides)

    if args.sweep:
        if "=" not in args.sweep:
            ap.error("--sweep 需要 KEY=V1,V2,...")
        key, raw_values = args.sweep.split("=", 1)
        if not hasattr(policy, key):
            ap.error(f"未知策略参数 {key!r}")
        print(f"{key:>22}  wake/天  中位间隔  表达/天  单日峰值  预算拦下")
        print("-" * 74)
        for raw in raw_values.split(","):
            pol = policy.with_overrides(**{key: _coerce(key, raw)})
            r = simulate(
                days=args.days,
                policy=pol,
                seed=args.seed,
                user_turns_per_day=args.user_turns_per_day,
                silent_rate=args.silent_rate,
                mmod=args.mmod,
                cold_start=not args.no_cold_start,
                csv_path=None,
            )
            iv = _intervals(r.wake_minutes)
            print(
                f"{raw:>22}  {len(r.wake_minutes) / r.days:7.1f}  "
                f"{_pct(iv, 0.5):8.0f}  {len(r.expressed_minutes) / r.days:7.1f}  "
                f"{_max_per_window(r.expressed_minutes, 1440):8d}  {r.budget_suppressed:8d}"
            )
        return 0

    r = simulate(
        days=args.days,
        policy=policy,
        seed=args.seed,
        user_turns_per_day=args.user_turns_per_day,
        silent_rate=args.silent_rate,
        mmod=args.mmod,
        cold_start=not args.no_cold_start,
        csv_path=args.csv_path,
    )
    report(r)
    if args.csv_path:
        print(f"\n逐分钟时间线已写入 {args.csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
