#!/usr/bin/env python3
"""持久化与 tick 入口测试（阶段 2）。

重点不是"SQL 能不能跑"，而是两件事：

1. 跑在数据库上的轨迹必须和纯内存逐分钟 step() 一模一样。
   这是阶段 1 重放测试的延伸：持久化不得改变节律。
2. 长间隔 / 重跑 / 并发这三类异常下，不会凭空冒出机会，也不会双插。

直接跑：python3 tests/test_storage.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wake.engine import init_state, report_agent_run, step  # noqa: E402
from wake.policy import DEFAULT_POLICY  # noqa: E402
from wake.storage import (  # noqa: E402
    LockBusy,
    StateConflict,
    connect,
    counts,
    ensure_schema,
    ensure_state,
    insert_opportunity,
    load_state,
    process_lock,
    save_state,
    write_transaction,
)
from wake.tick import do_report_run, run_tick  # noqa: E402

P = DEFAULT_POLICY


class _TempDB:
    """每个测试一个干净的临时库。"""

    def __enter__(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "kli_wake.db")
        self.conn = connect(self.path)
        ensure_schema(self.conn)
        return self

    def __exit__(self, *exc):
        self.conn.close()
        self._dir.cleanup()
        return False

    def reopen(self):
        """模拟进程退出重进——形态 B 每分钟都在干这件事。"""
        self.conn.close()
        self.conn = connect(self.path)
        ensure_schema(self.conn)
        return self.conn


def test_first_tick_initializes_then_never_again():
    """首次 tick 建状态，之后永远读旧状态。重新初始化 = 重抽节律，是严重 bug。"""
    with _TempDB() as db:
        s1 = run_tick(db.conn, 1000, P, seed=42)
        assert s1["initialized"]
        st = load_state(db.conn)
        assert st is not None and st.minute == 1000
        assert not st.warmup_complete, "首次应该是 cold start"

        db.reopen()
        s2 = run_tick(db.conn, 1001, P, seed=99)
        assert not s2["initialized"], "第二次绝不能重新初始化"
        st2 = load_state(db.conn)
        assert st2.cycle_id == st.cycle_id or st2.cycle_seq >= st.cycle_seq


def test_db_trajectory_matches_pure_memory():
    """逐分钟 tick（每次都重开连接）必须和纯内存 step() 逐分钟一致。

    这是整个阶段 2 最重要的一条：落库 / JSON 序列化 / 进程重建都不得扰动轨迹。
    """
    start = 500_000
    minutes = 240

    with _TempDB() as db:
        run_tick(db.conn, start, P, seed=7)
        seeded = load_state(db.conn)

        db_wakes = []
        for i in range(1, minutes + 1):
            db.reopen()  # 每分钟一个新进程
            summary = run_tick(db.conn, start + i, P)
            db_wakes.extend(summary["opportunities"])
        final_db = load_state(db.conn)

    cur = seeded
    mem_wakes = 0
    for _ in range(minutes):
        cur, wake = step(cur, P)
        if wake:
            mem_wakes += 1

    assert final_db.minute == cur.minute
    assert abs(final_db.drive - cur.drive) < 1e-12, "D 轨迹分叉"
    assert abs(final_db.tone - cur.tone) < 1e-12, "T 轨迹分叉"
    assert abs(final_db.drift - cur.drift) < 1e-12, "X 轨迹分叉"
    assert abs(final_db.hazard - cur.hazard) < 1e-12, "hazard 分叉"
    assert final_db.theta == cur.theta, "Θ 分叉（RNG 消耗顺序不一致）"
    assert final_db.cycle_seq == cur.cycle_seq
    assert len(db_wakes) == mem_wakes, f"wake 数不等: {len(db_wakes)} vs {mem_wakes}"


def test_catchup_in_one_tick_matches_minute_by_minute():
    """一次 tick 距上次 10 分钟（≤ max_gap），结果应等于跑 10 次每次 1 分钟。

    cron 漏跑几分钟是常态，不能因此改变节律。
    """
    start = 700_000
    gap = 10
    assert gap <= P.max_gap_minutes

    with _TempDB() as a, _TempDB() as b:
        run_tick(a.conn, start, P, seed=13)
        run_tick(b.conn, start, P, seed=13)

        sa = run_tick(a.conn, start + gap, P)
        for i in range(1, gap + 1):
            sb = run_tick(b.conn, start + i, P)

        fa, fb = load_state(a.conn), load_state(b.conn)
        assert fa.minute == fb.minute
        assert abs(fa.drive - fb.drive) < 1e-12
        assert abs(fa.hazard - fb.hazard) < 1e-12
        assert fa.theta == fb.theta
        assert sa["minutes_advanced"] == gap
        assert counts(a.conn)["opportunities"] == counts(b.conn)["opportunities"]


def test_long_gap_suppresses_and_logs():
    """停机 8h 后的一次 tick：不补发机会，记 suppressed，但状态照常演化。"""
    with _TempDB() as db:
        run_tick(db.conn, 800_000, P, seed=3)
        do_report_run(db.conn, "user_turn", 800_000, P)
        low = load_state(db.conn).drive
        assert low < P.d_mean

        before = counts(db.conn)["opportunities"]
        summary = run_tick(db.conn, 800_000 + 8 * 60, P)
        after = counts(db.conn)

        assert after["opportunities"] == before, "停机期间错过的自发唤醒不能补发"
        assert after["suppressed"] == 1
        kinds = [e["kind"] for e in summary["events"]]
        assert "suppressed_spontaneous" in kinds

        st = load_state(db.conn)
        assert st.hazard == 0.0, "cycle 必须归零"
        assert abs(st.drive - P.d_mean) < 1e-6, f"D 没有演化：{st.drive}"


def test_replaying_same_minute_is_idempotent():
    """同一分钟重跑 tick（cron 重叠 / 手动多敲一次）不能推进状态也不能双插。"""
    with _TempDB() as db:
        run_tick(db.conn, 900_000, P, seed=21)
        run_tick(db.conn, 900_001, P)
        st1 = load_state(db.conn)
        c1 = counts(db.conn)

        for _ in range(5):
            summary = run_tick(db.conn, 900_001, P)
            assert summary["minutes_advanced"] == 0

        st2 = load_state(db.conn)
        assert st2.minute == st1.minute
        assert st2.drive == st1.drive
        assert st2.hazard == st1.hazard
        assert st2.theta == st1.theta
        assert counts(db.conn)["opportunities"] == c1["opportunities"]


def test_opportunity_id_is_deterministic():
    """机会 id 由 (minute, cycle_seq) 确定，PRIMARY KEY 保证幂等。"""
    with _TempDB() as db:
        with write_transaction(db.conn):
            first = insert_opportunity(
                db.conn, "wk_test_000001", "spontaneous", "cyc_x", 1234
            )
            second = insert_opportunity(
                db.conn, "wk_test_000001", "spontaneous", "cyc_x", 1234
            )
        assert first is True
        assert second is False, "重复 id 应该被忽略而不是报错或双插"
        assert counts(db.conn)["opportunities"] == 1


def test_agent_run_kick_persists_and_keeps_cycle():
    """kick D 落库，但绝不触碰 hazard / theta / cycle，也不消耗 RNG。"""
    with _TempDB() as db:
        run_tick(db.conn, 950_000, P, seed=5)
        before = load_state(db.conn)

        do_report_run(db.conn, "direct_wake", 950_000, P)
        after = load_state(db.conn)

        assert abs(after.drive - (before.drive - P.k_run)) < 1e-12
        assert after.hazard == before.hazard
        assert after.theta == before.theta
        assert after.cycle_id == before.cycle_id
        assert after.rng_state == before.rng_state, "kick 不应消耗 RNG"
        assert after.runs_total == before.runs_total + 1


def test_optimistic_lock_rejects_stale_write():
    """state_version 对不上时必须报错，而不是默默覆盖。"""
    with _TempDB() as db:
        run_tick(db.conn, 960_000, P, seed=8)
        st = load_state(db.conn)

        with write_transaction(db.conn):
            save_state(db.conn, report_agent_run(st, P), st.state_version)

        try:
            with write_transaction(db.conn):
                save_state(db.conn, report_agent_run(st, P), st.state_version)
        except StateConflict:
            pass
        else:
            raise AssertionError("陈旧 state_version 的写入应该被拒绝")


def test_process_lock_is_exclusive():
    """第二个 tick 进程拿不到锁应该直接退出，不排队。"""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "kli_wake.db")
        with process_lock(path):
            try:
                with process_lock(path):
                    raise AssertionError("应该拿不到锁")
            except LockBusy:
                pass
        # 释放后能再拿
        with process_lock(path):
            pass


def test_ensure_state_is_race_safe():
    """两个进程同时首次启动时，ensure_state 只能有一个赢，另一个读到同一个状态。"""
    with _TempDB() as db:
        a = init_state(1_000_000, P, seed=1)
        b = init_state(1_000_000, P, seed=2)
        with write_transaction(db.conn):
            got_a, created_a = ensure_state(db.conn, a)
        with write_transaction(db.conn):
            got_b, created_b = ensure_state(db.conn, b)
        assert created_a is True
        assert created_b is False, "第二次不能覆盖已有状态"
        assert got_b.theta == got_a.theta, "应该读到已有状态，而不是自己那份"


def test_snapshots_recorded_for_curves():
    """每 tick 一行快照，后面靠它画 λ(t) 曲线调参。"""
    with _TempDB() as db:
        run_tick(db.conn, 1_100_000, P, seed=4)
        for i in range(1, 31):
            run_tick(db.conn, 1_100_000 + i, P)
        n = counts(db.conn)["snapshots"]
        assert n == 31, f"快照数不对：{n}"

        rows = db.conn.execute(
            "SELECT minute, lambda_per_hour FROM state_snapshots ORDER BY minute"
        ).fetchall()
        assert rows[0]["minute"] == 1_100_000
        assert all(
            P.lambda_min_per_hour <= r["lambda_per_hour"] <= P.lambda_max_per_hour
            for r in rows
        )


def test_expired_opportunities_are_not_replayed():
    """没人取的机会过期就丢，不排队。阶段 2 没有 Agent，所以全部会过期——这是预期的。"""
    with _TempDB() as db:
        run_tick(db.conn, 1_200_000, P, seed=17)
        for i in range(1, 121):
            run_tick(db.conn, 1_200_000 + i, P)
        c = counts(db.conn)
        if c["opportunities"] > 0:
            assert c["opportunities_expired"] > 0, "超时未处理的机会应该被标 EXPIRED"


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
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
