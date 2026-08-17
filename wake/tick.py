#!/usr/bin/env python3
"""tick 入口：形态 B 的心跳目标。

没有常驻进程。systemd timer / cron 每分钟调一次，进程跑完就退。

    python3 -m wake.tick                    推进到当前整分钟
    python3 -m wake.tick status              看当前状态和计数
    python3 -m wake.tick report-run --source user_turn
    python3 -m wake.tick export --out /tmp/wake.csv --days 3

这一层只负责"把节律推进并把机会落库"。没有 Dispatcher，没有 Agent，
没有任何东西会发消息。机会就存在 wake_opportunities 里等着过期——这是预期的。

先让它安静地跑几天，看节律在真实时间里长什么样，再接 Agent。
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
import time

from .engine import (
    advance,
    diagnostics,
    epoch_minute,
    init_state,
    lambda_of_state,
    report_agent_run,
)
from .policy import DEFAULT_POLICY, WakePolicy
from .storage import (
    DEFAULT_DB_PATH,
    LockBusy,
    connect,
    counts,
    ensure_schema,
    ensure_state,
    expire_stale_opportunities,
    insert_event,
    insert_opportunity,
    insert_snapshot,
    insert_wake_event,
    load_state,
    minute_to_iso,
    opportunity_id,
    process_lock,
    prune_snapshots,
    save_state,
    write_transaction,
)

# 快照保留天数。每分钟一行，90 天约 13 万行，几 MB，无所谓。
SNAPSHOT_RETENTION_DAYS = 90


def _mmod_now() -> float:
    """当前的主观调制因子。

    阶段 2 没有 modulation 层，恒为 1.0。以后 EB / 记忆 / OpenLoop 接进来时，
    在这里读聚合后的 Mmod（log-space 聚合 + 统一 clamp，带 TTL 衰减）。
    注意它只能调制 λ，不能直接 wake_now。
    """
    return 1.0


def _opp_id_for(event) -> str:
    """从 cycle_id 尾部取 seq，凑出确定性的机会 id。

    cycle_id 形如 cyc_{minute:09d}_{seq:06d}，见 engine._cycle_id()。
    """
    seq = int(event.cycle_id.rsplit("_", 1)[-1])
    return opportunity_id(event.minute, seq)


def run_tick(
    conn: sqlite3.Connection,
    target_minute: int,
    policy: WakePolicy = DEFAULT_POLICY,
    seed: int | None = None,
    mmod: float | None = None,
) -> dict:
    """把状态推进到 target_minute，落库。整个过程在一个事务里。

    target_minute 作为参数传入而不是内部读时间，是为了测试能逐分钟驱动。
    """
    m = _mmod_now() if mmod is None else mmod
    summary: dict = {
        "target_minute": target_minute,
        "initialized": False,
        "minutes_advanced": 0,
        "opportunities": [],
        "events": [],
        "expired": 0,
        "pruned": 0,
    }

    with write_transaction(conn):
        state = load_state(conn)

        if state is None:
            # 只在完全没有状态时初始化。重启读旧状态，绝不重新抽节律。
            # 走 ensure_state 而不是裸 insert：两个进程同时首次启动时
            # 都会看到 None，后一个不能覆盖前一个。
            if seed is None:
                seed = int.from_bytes(os.urandom(8), "big")
            candidate = init_state(
                target_minute, policy, seed=seed, cold_start=True
            )
            state, created = ensure_state(conn, candidate)
            if created:
                insert_event(
                    conn,
                    "activation_initialized",
                    target_minute,
                    state.cycle_id,
                    {
                        "policy_version": policy.version,
                        "policy_fingerprint": policy.fingerprint(),
                        "seed": str(seed),
                        "cold_start": True,
                        "warmup_minutes": policy.warmup_minutes,
                    },
                )
                insert_snapshot(
                    conn, state, lambda_of_state(state, policy, m), m, wake=False
                )
                summary["initialized"] = True
                summary["state"] = diagnostics(state, policy, m)
                return summary
            # 竞态输了：另一个进程刚建好状态。用它的状态继续走正常路径。

        expected_version = state.state_version
        nxt, events = advance(state, target_minute, policy, m)
        summary["minutes_advanced"] = max(0, nxt.minute - state.minute)

        for ev in events:
            insert_wake_event(conn, ev)
            summary["events"].append({"kind": ev.kind, "minute": ev.minute})
            if ev.kind == "spontaneous_wake":
                # cycle_seq 已经是新 cycle 的，但对 (minute, seq) 而言仍然唯一且可重建。
                opp_id = _opp_id_for(ev)
                created = insert_opportunity(
                    conn,
                    opp_id=opp_id,
                    source="spontaneous",
                    cycle_id=ev.cycle_id,
                    created_minute=ev.minute,
                    detail={"activation_minute": ev.minute},
                )
                if created:
                    summary["opportunities"].append(opp_id)

        if nxt.minute != state.minute or nxt.state_version != expected_version:
            save_state(conn, nxt, expected_version)

        if summary["minutes_advanced"] > 0:
            insert_snapshot(
                conn,
                nxt,
                lambda_of_state(nxt, policy, m),
                m,
                wake=any(e.kind == "spontaneous_wake" for e in events),
            )

        summary["expired"] = expire_stale_opportunities(conn, target_minute)

        if target_minute % 60 == 0:
            summary["pruned"] = prune_snapshots(
                conn, target_minute - SNAPSHOT_RETENTION_DAYS * 1440
            )

        summary["state"] = diagnostics(nxt, policy, m)

    return summary


def do_report_run(
    conn: sqlite3.Connection,
    source: str,
    target_minute: int,
    policy: WakePolicy = DEFAULT_POLICY,
    mmod: float | None = None,
) -> dict:
    """报告一次真实 Agent Run：先推进到当前，再 kick D。

    任何真实运行都要调：UserTurn、Direct Wake、Spontaneous Wake。
    漏掉 UserTurn 是最容易犯的——那会让 D 偏高，自发唤醒比设计频繁。

    它不触碰 hazard / theta / cycle，也不消耗 RNG。
    """
    m = _mmod_now() if mmod is None else mmod
    run_tick(conn, target_minute, policy, mmod=m)

    with write_transaction(conn):
        state = load_state(conn)
        if state is None:
            raise RuntimeError("没有 ActivationState，先跑一次 tick")
        expected_version = state.state_version
        before = state.drive
        nxt = report_agent_run(state, policy)
        save_state(conn, nxt, expected_version)
        insert_event(
            conn,
            "agent_run",
            nxt.minute,
            nxt.cycle_id,
            {
                "source": source,
                "drive_before": round(before, 4),
                "drive_after": round(nxt.drive, 4),
                "k_run": policy.k_run,
                "runs_total": nxt.runs_total,
            },
        )
        return diagnostics(nxt, policy, m)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _cmd_tick(args) -> int:
    conn = connect(args.db)
    try:
        ensure_schema(conn)
        target = args.minute if args.minute is not None else epoch_minute(time.time())
        summary = run_tick(conn, target, DEFAULT_POLICY, seed=args.seed)
    finally:
        conn.close()

    if args.quiet:
        return 0
    if summary["initialized"]:
        print(f"initialized  minute={summary['target_minute']} 已建立 ActivationState")
        return 0
    st = summary["state"]
    parts = [
        f"minute={st['minute']}",
        f"+{summary['minutes_advanced']}min",
        f"D={st['drive']:.3f}",
        f"T={st['tone']:.3f}",
        f"X={st['drift']:+.3f}",
        f"λ={st['lambda_per_hour']:.2f}/h",
        f"H/Θ={st['hazard']:.3f}/{st['theta']:.3f}",
    ]
    if summary["opportunities"]:
        parts.append("wake=" + ",".join(summary["opportunities"]))
    for ev in summary["events"]:
        if ev["kind"] != "spontaneous_wake":
            parts.append(ev["kind"])
    print("  ".join(parts))
    return 0


def _cmd_status(args) -> int:
    conn = connect(args.db)
    try:
        ensure_schema(conn)
        state = load_state(conn)
        if state is None:
            print("还没有 ActivationState。跑一次 `python3 -m wake.tick` 就会初始化。")
            return 0
        dg = diagnostics(state, DEFAULT_POLICY, _mmod_now())
        now = epoch_minute(time.time())
        c = counts(conn)
    finally:
        conn.close()

    print(f"db                {os.path.abspath(args.db)}")
    print(f"policy            {dg['policy_version']}")
    print(
        f"lastEvaluatedAt   {dg['minute']}  ({minute_to_iso(dg['minute'])})  "
        f"距今 {now - dg['minute']} 分钟"
    )
    print(
        f"D / T / X         {dg['drive']:.3f} / {dg['tone']:.3f} / {dg['drift']:+.3f}"
    )
    print(
        f"λ(t)              {dg['lambda_per_hour']:.2f}/h     "
        f"H/Θ {dg['hazard']:.3f} / {dg['theta']:.3f}（还差 {dg['hazard_remaining']:.3f}）"
    )
    print(f"cycle             {dg['cycle_id']}  已持续 {dg['cycle_age_minutes']} 分钟")
    print(
        f"warmup            {'complete' if dg['warmup_complete'] else 'active'}     "
        f"stateVersion {dg['state_version']}     AgentRun 累计 {dg['runs_total']}"
    )
    print(
        f"opportunities     共 {c['opportunities']}（CREATED {c['opportunities_created']}"
        f" / EXPIRED {c['opportunities_expired']}）"
    )
    print(
        f"events            共 {c['events']}（suppressed {c['suppressed']}）     "
        f"snapshots {c['snapshots']}"
    )
    if now - dg["minute"] > DEFAULT_POLICY.max_gap_minutes:
        print(
            f"\n注意：距今超过 {DEFAULT_POLICY.max_gap_minutes} 分钟，下次 tick 会走长间隔恢复"
            "（状态照常演化，hazard 丢弃，cycle 重抽，不补发）。"
        )
    return 0


def _cmd_report_run(args) -> int:
    conn = connect(args.db)
    try:
        ensure_schema(conn)
        target = args.minute if args.minute is not None else epoch_minute(time.time())
        dg = do_report_run(conn, args.source, target, DEFAULT_POLICY)
    finally:
        conn.close()
    print(
        f"agent_run source={args.source}  D={dg['drive']:.3f}  "
        f"λ={dg['lambda_per_hour']:.2f}/h  runs_total={dg['runs_total']}"
    )
    return 0


def _cmd_export(args) -> int:
    conn = connect(args.db)
    try:
        ensure_schema(conn)
        since = epoch_minute(time.time()) - int(args.days * 1440)
        rows = conn.execute(
            "SELECT minute, drive, tone, drift, lambda_per_hour, hazard, theta, "
            "mmod, cycle_id, wake FROM state_snapshots WHERE minute >= ? "
            "ORDER BY minute",
            (since,),
        ).fetchall()
    finally:
        conn.close()

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "minute",
                "iso",
                "drive",
                "tone",
                "drift",
                "lambda_per_hour",
                "hazard",
                "theta",
                "mmod",
                "cycle_id",
                "wake",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["minute"],
                    minute_to_iso(r["minute"]),
                    r["drive"],
                    r["tone"],
                    r["drift"],
                    r["lambda_per_hour"],
                    r["hazard"],
                    r["theta"],
                    r["mmod"],
                    r["cycle_id"],
                    r["wake"],
                ]
            )
    print(f"{len(rows)} 行 → {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Kli Wake Runtime tick（形态 B：无常驻进程）"
    )
    ap.add_argument("--db", default=DEFAULT_DB_PATH, help=f"默认 {DEFAULT_DB_PATH}")
    sub = ap.add_subparsers(dest="cmd")

    p_tick = sub.add_parser("tick", help="推进到当前整分钟（默认子命令）")
    p_tick.add_argument("--minute", type=int, default=None, help="指定目标分钟，测试用")
    p_tick.add_argument("--seed", type=int, default=None, help="仅首次初始化时使用")
    p_tick.add_argument("-q", "--quiet", action="store_true")
    p_tick.set_defaults(func=_cmd_tick)

    p_status = sub.add_parser("status", help="看当前状态")
    p_status.set_defaults(func=_cmd_status)

    p_run = sub.add_parser("report-run", help="报告一次真实 Agent Run（kick D）")
    p_run.add_argument(
        "--source",
        default="user_turn",
        help="user_turn / direct_wake / spontaneous_wake",
    )
    p_run.add_argument("--minute", type=int, default=None)
    p_run.set_defaults(func=_cmd_report_run)

    p_exp = sub.add_parser("export", help="导出快照 CSV，画曲线用")
    p_exp.add_argument("--out", default="wake_snapshots.csv")
    p_exp.add_argument("--days", type=float, default=7.0)
    p_exp.set_defaults(func=_cmd_export)

    args = ap.parse_args(argv)
    if not hasattr(args, "func"):
        # 不带子命令时默认 tick，方便 cron 直接写 `python3 -m wake.tick -q`
        args = ap.parse_args((argv or []) + ["tick"])

    # status / export 是只读的，不抢锁；写路径必须拿锁。
    if args.func in (_cmd_status, _cmd_export):
        return args.func(args)

    try:
        with process_lock(args.db):
            return args.func(args)
    except LockBusy as exc:
        if not getattr(args, "quiet", False):
            print(f"skip: {exc}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
