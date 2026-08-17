#!/usr/bin/env python3
"""确定性重放测试。

这组测试比 λ 公式的单测重要得多：公式写错了看曲线就能发现，
轨迹在重启处分叉却几乎不可能被肉眼发现，只会表现为"节律有时候莫名其妙变了"。

直接跑：python tests/test_replay.py
pytest 也能收（纯 assert，无 fixture 依赖）。
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wake.engine import (  # noqa: E402
    ActivationState,
    advance,
    init_state,
    lambda_per_hour,
    report_agent_run,
    step,
)
from wake.policy import DEFAULT_POLICY  # noqa: E402

P = DEFAULT_POLICY


def _run(state, minutes):
    """逐分钟跑，返回 (最终状态, wake 分钟列表, 逐分钟轨迹)。"""
    trace, wakes = [], []
    cur = state
    for _ in range(minutes):
        cur, wake = step(cur, P)
        trace.append((cur.minute, cur.drive, cur.tone, cur.drift, cur.hazard, cur.theta))
        if wake:
            wakes.append(cur.minute)
    return cur, wakes, trace


def test_restart_matches_continuous_run():
    """重启续跑必须和连续运行逐分钟完全一致（同 seed + 同 state + 同 policy）。"""
    s0 = init_state(0, P, seed=42)

    _, wakes_a, trace_a = _run(s0, 720)  # 连续 12h

    mid, wakes_b1, trace_b1 = _run(s0, 300)
    # 模拟落库再读回：JSON round trip 必须完整保留 RNG 状态
    revived = ActivationState.from_json(json.loads(json.dumps(mid.to_json())))
    assert revived == mid, "JSON round trip 改变了状态"
    _, wakes_b2, trace_b2 = _run(revived, 420)

    assert trace_a == trace_b1 + trace_b2, "重启处轨迹分叉了"
    assert wakes_a == wakes_b1 + wakes_b2, f"wake 时刻不一致: {wakes_a} vs {wakes_b1 + wakes_b2}"


def test_short_gap_accumulates_hazard():
    """空档 ≤ max_gap_minutes：正常演化并累积 hazard，等价于逐分钟 step。"""
    s0 = init_state(0, P, seed=7)
    gap = P.max_gap_minutes
    via_advance, events = advance(s0, gap, P)
    via_steps, _, _ = _run(s0, gap)
    assert via_advance == via_steps
    assert not [e for e in events if e.kind == "suppressed_spontaneous"]


def test_long_gap_suppresses_but_still_evolves():
    """长间隔恢复的三件事必须互相独立：

    状态照常演化 / hazard 增量丢弃 / cycle 作废重抽。
    最常见的实现错误是"直接重锚"——把停机前的 D 原样冻回来。
    """
    s0 = init_state(0, P, seed=11)
    # 先把 D 压低，这样"是否演化"才能被观测到
    s0 = report_agent_run(report_agent_run(s0, P), P)
    low_drive = s0.drive
    assert low_drive < P.d_mean

    gap = 8 * 60  # 停机 8h
    nxt, events = advance(s0, s0.minute + gap, P)

    kinds = [e.kind for e in events]
    assert "spontaneous_wake" not in kinds, "停机期间错过的自发唤醒不能补发"
    assert kinds.count("suppressed_spontaneous") == 1

    assert nxt.minute == s0.minute + gap
    assert nxt.hazard == 0.0, "cycle 必须归零，不能带着停机期间累积的 H"
    assert nxt.cycle_id != s0.cycle_id, "cycle 必须作废重抽"
    assert nxt.theta != s0.theta

    # 8h 远超 τD=12min，D 必须已经回归到接近 μD，而不是停在 low_drive
    assert abs(nxt.drive - P.d_mean) < 1e-6, f"D 没有演化: {nxt.drive}"
    assert nxt.drive > low_drive


def test_clock_regression_is_ignored():
    """时钟回拨不能让状态倒退，只记 anomaly。"""
    s0 = init_state(1000, P, seed=3)
    same, ev_same = advance(s0, 1000, P)
    assert same == s0 and ev_same == []

    back, ev_back = advance(s0, 900, P)
    assert back == s0
    assert [e.kind for e in ev_back] == ["clock_regression"]


def test_agent_run_never_touches_cycle():
    """Direct Wake / UserTurn 只 kick D，绝不干扰自发节律，也不消耗 RNG。"""
    s0 = init_state(0, P, seed=5)
    s1 = report_agent_run(s0, P)
    assert abs(s1.drive - (s0.drive - P.k_run)) < 1e-12
    assert s1.hazard == s0.hazard
    assert s1.theta == s0.theta
    assert s1.cycle_id == s0.cycle_id
    assert s1.rng_state == s0.rng_state, "kick 不应消耗随机数，否则轨迹会分叉"
    assert s1.runs_total == s0.runs_total + 1


def test_drive_kick_respects_bounds():
    s = init_state(0, P, seed=5)
    for _ in range(50):
        s = report_agent_run(s, P)
    assert s.drive == P.d_min


def test_lambda_bounds_and_monotonicity():
    lo = lambda_per_hour(P.d_min, P.t_min, P.x_min, P)
    mid = lambda_per_hour(P.d_mean, P.t_mean, P.x_mean, P)
    hi = lambda_per_hour(P.d_max, P.t_max, P.x_max, P)
    assert lo < mid < hi
    assert abs(mid - P.lambda_base_per_hour) < 1e-9, "中性状态下 λ 应等于 λθ"
    assert P.lambda_min_per_hour <= lo
    assert hi <= P.lambda_max_per_hour

    # Mmod 只调制，且被 clamp 死
    assert lambda_per_hour(P.d_mean, P.t_mean, 0.0, P, mmod=100.0) <= P.lambda_max_per_hour
    assert lambda_per_hour(P.d_mean, P.t_mean, 0.0, P, mmod=0.001) >= P.lambda_min_per_hour


def test_baseline_density_in_expected_range():
    """基线密度 sanity check：中性起步、无 UserTurn 时大致 15~45 次/天。

    文档给的参考是"半小时内自然醒一次很正常"。这个测试防止手滑把 λθ 或 β 改错一个量级，
    不是精度断言。
    """
    s = init_state(0, P, seed=99, cold_start=False)
    wakes = 0
    for _ in range(30 * 1440):
        s, wake = step(s, P)
        if wake:
            wakes += 1
            s = report_agent_run(s, P)  # 自发唤醒也是真实 Run
    per_day = wakes / 30
    assert 15 <= per_day <= 45, f"基线密度偏离预期: {per_day:.1f} 次/天"


def test_warmup_completes_once():
    s = init_state(0, P, seed=1)
    assert not s.warmup_complete
    assert s.drive < P.d_mean and s.tone < P.t_mean
    for _ in range(P.warmup_minutes + 1):
        s, _ = step(s, P)
    assert s.warmup_complete


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
