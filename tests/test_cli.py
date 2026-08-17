#!/usr/bin/env python3
"""CLI 入口测试。

存在的理由很具体：`python3 -m wake.tick -q`（systemd 单元里就这么写的）
曾经直接 exit(2)——-q 定义在 tick 子命令上，顶层 parser 不认识，
argparse 在解析阶段就退了，“不带子命令默认 tick”的兵底根本轮不到。

引擎再对，入口跑不起来也是白搭。

直接跑：python3 tests/test_cli.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wake.storage import connect, counts, ensure_schema, load_state  # noqa: E402
from wake.tick import _normalize_argv, main  # noqa: E402


class _TempDBPath:
    def __enter__(self) -> str:
        self._dir = tempfile.TemporaryDirectory()
        return os.path.join(self._dir.name, "kli_wake.db")

    def __exit__(self, *exc):
        self._dir.cleanup()
        return False


def test_normalize_inserts_tick():
    assert _normalize_argv([]) == ["tick"]
    assert _normalize_argv(["-q"]) == ["tick", "-q"]
    assert _normalize_argv(["--db", "/x/y.db", "-q"]) == [
        "--db",
        "/x/y.db",
        "tick",
        "-q",
    ]
    assert _normalize_argv(["--db=/x/y.db", "-q"]) == ["--db=/x/y.db", "tick", "-q"]


def test_normalize_leaves_explicit_subcommands_alone():
    for cmd in ("tick", "status", "report-run", "export"):
        assert _normalize_argv([cmd]) == [cmd]
    assert _normalize_argv(["--db", "/x", "status"]) == ["--db", "/x", "status"]
    assert _normalize_argv(["tick", "-q"]) == ["tick", "-q"]


def test_bare_quiet_flag_runs_tick():
    """systemd 单元里的写法必须能跑。这次真把部署坑了：exit 2/INVALIDARGUMENT。"""
    with _TempDBPath() as path:
        rc = main(["--db", path, "-q"])
        assert rc == 0, f"退码应为 0，实际 {rc}"
        conn = connect(path)
        try:
            ensure_schema(conn)
            st = load_state(conn)
            assert st is not None, "首次 tick 应该建立了 ActivationState"
        finally:
            conn.close()


def test_bare_no_args_runs_tick():
    with _TempDBPath() as path:
        assert main(["--db", path]) == 0
        conn = connect(path)
        try:
            assert load_state(conn) is not None
        finally:
            conn.close()


def test_status_and_export_do_not_need_state():
    """status / export 在空库上不能崩，也不应该抢锁。"""
    with _TempDBPath() as path:
        assert main(["--db", path, "status"]) == 0
        out = os.path.join(os.path.dirname(path), "out.csv")
        assert main(["--db", path, "export", "--out", out, "--days", "1"]) == 0
        assert os.path.exists(out)


def test_explicit_minute_advances_and_reports_run():
    """逐分钟驱动 + report-run，走一遍完整 CLI 路径。"""
    with _TempDBPath() as path:
        base = 2_000_000
        assert main(["--db", path, "tick", "--minute", str(base), "--seed", "5"]) == 0
        for i in range(1, 6):
            assert main(["--db", path, "tick", "--minute", str(base + i), "-q"]) == 0

        conn = connect(path)
        try:
            st = load_state(conn)
            assert st.minute == base + 5
            assert counts(conn)["snapshots"] == 6
            before = st.drive
        finally:
            conn.close()

        assert (
            main(
                [
                    "--db",
                    path,
                    "report-run",
                    "--source",
                    "user_turn",
                    "--minute",
                    str(base + 5),
                ]
            )
            == 0
        )

        conn = connect(path)
        try:
            st2 = load_state(conn)
            assert st2.drive < before, "report-run 应该 kick 低 D"
            assert st2.runs_total == 1
        finally:
            conn.close()


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
        except SystemExit as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: 意外 exit({e.code})")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
