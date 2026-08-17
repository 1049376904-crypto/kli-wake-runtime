"""持久化层：SQLite 作为唯一权威状态。

形态 B：没有常驻进程。外部心跳（systemd timer / cron）每分钟调一次 tick，
进程跑完就退。真正的节律活在这个数据库里，心跳只是给它推进的机会。

这样做的好处是不存在"进程活着但卡住了"这类状态：cron 漏跑、机器重启、
手动 stop，全部走同一条恢复路径，而这条路径已经有测试覆盖。

代价是跮了 monotonic clock：进程每分钟重建，monotonic 计数器跟着残废，
跳不过进程边界。所以 dt 只能靠 wall clock 算，时钟回拨的防护只剩一层：
engine.advance() 拒绝往回走，并记 clock_regression。NTP 小幅矫正不会出问题，
手动把系统时间往前拨大步会被当成长间隔恢复（不补发，cycle 作废）。

写入并发由两层守：文件锁（防两个 tick 进程重叠）+ BEGIN IMMEDIATE 事务
加 state_version 乐观校验（防半更新）。单用户单 Agent，不需要 lease。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from .engine import ActivationState, WakeEvent, minute_to_epoch_seconds

SCHEMA_VERSION = 1

DEFAULT_DB_PATH = os.environ.get(
    "KLI_WAKE_DB", os.path.expanduser("~/.local/share/kli-wake/kli_wake.db")
)


class StateConflict(RuntimeError):
    """state_version 对不上：有另一个写者插进来了。本次 tick 应该整体回滚。"""


class LockBusy(RuntimeError):
    """另一个 tick 正在跑。直接退出，不排队——下一分钟还会有机会。"""


DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- 单行权威状态（id 永远为 1）。没有这行才初始化；重启读旧状态，绝不重新初始化。
CREATE TABLE IF NOT EXISTS activation_state (
    id                    INTEGER PRIMARY KEY CHECK (id = 1),
    minute                INTEGER NOT NULL,
    drive                 REAL    NOT NULL,
    tone                  REAL    NOT NULL,
    drift                 REAL    NOT NULL,
    hazard                REAL    NOT NULL,
    theta                 REAL    NOT NULL,
    cycle_id              TEXT    NOT NULL,
    cycle_started_minute  INTEGER NOT NULL,
    cycle_seq             INTEGER NOT NULL,
    rng_state             TEXT    NOT NULL,
    state_version         INTEGER NOT NULL,
    policy_version        TEXT    NOT NULL,
    warmup_started_minute INTEGER NOT NULL,
    warmup_complete       INTEGER NOT NULL,
    runs_total            INTEGER NOT NULL,
    updated_at            TEXT    NOT NULL
);

-- 每 tick 一行，用来画 λ(t) / D / T / X 曲线。调参靠看这个，不靠感觉。
-- 只写日志，绝不进 Agent 上下文。
CREATE TABLE IF NOT EXISTS state_snapshots (
    minute          INTEGER PRIMARY KEY,
    drive           REAL NOT NULL,
    tone            REAL NOT NULL,
    drift           REAL NOT NULL,
    lambda_per_hour REAL NOT NULL,
    hazard          REAL NOT NULL,
    theta           REAL NOT NULL,
    mmod            REAL NOT NULL,
    cycle_id        TEXT NOT NULL,
    wake            INTEGER NOT NULL DEFAULT 0
);

-- 运行机会。阶段 2 只创建（status=CREATED），没有 Dispatcher 也没有 Agent。
-- id 由 minute + cycle_seq 确定，UNIQUE 保证同一分钟重跑 tick 不会双插。
CREATE TABLE IF NOT EXISTS wake_opportunities (
    id            TEXT PRIMARY KEY,
    source        TEXT    NOT NULL,
    cycle_id      TEXT    NOT NULL,
    created_minute INTEGER NOT NULL,
    expires_minute INTEGER NOT NULL,
    priority      TEXT    NOT NULL DEFAULT 'normal',
    replay        TEXT    NOT NULL DEFAULT 'none',
    status        TEXT    NOT NULL DEFAULT 'CREATED',
    created_at    TEXT    NOT NULL,
    detail        TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_opp_status ON wake_opportunities(status, created_minute);

-- 事件日志：suppressed_spontaneous / clock_regression / catchup_truncated /
-- agent_run / init 等。出问题时靠它回溯。
CREATE TABLE IF NOT EXISTS wake_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT    NOT NULL,
    minute     INTEGER NOT NULL,
    cycle_id   TEXT    NOT NULL,
    detail     TEXT    NOT NULL DEFAULT '{}',
    created_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_kind ON wake_events(kind, minute);
"""


# --------------------------------------------------------------------------
# 连接
# --------------------------------------------------------------------------

def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """打开（必要时创建）数据库。isolation_level=None：手动管事务。"""
    directory = os.path.dirname(os.path.abspath(db_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    conn.execute(
        "INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )


@contextmanager
def write_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """BEGIN IMMEDIATE：立刻拿写锁，不等到第一条 UPDATE。

    状态更新和 WakeOpportunity 创建必须在同一事务里，否则会出现
    "机会写了但状态没推进"或反之的半更新。
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


# --------------------------------------------------------------------------
# 进程锁
# --------------------------------------------------------------------------

@dataclass
class _Lock:
    fd: int
    path: str


@contextmanager
def process_lock(db_path: str = DEFAULT_DB_PATH) -> Iterator[None]:
    """非阻塞文件锁，防两个 tick 进程重叠。

    拿不到就抛 LockBusy，调用方应该直接退出而不是排队：
    自发 Wake 是机会不是欠账，下一分钟还会再来。
    """
    import fcntl

    lock_path = os.path.abspath(db_path) + ".lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise LockBusy(f"另一个 tick 正在运行（{lock_path}）") from exc
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


# --------------------------------------------------------------------------
# 状态读写
# --------------------------------------------------------------------------

_STATE_COLUMNS = (
    "minute",
    "drive",
    "tone",
    "drift",
    "hazard",
    "theta",
    "cycle_id",
    "cycle_started_minute",
    "cycle_seq",
    "rng_state",
    "state_version",
    "policy_version",
    "warmup_started_minute",
    "warmup_complete",
    "runs_total",
)


def load_state(conn: sqlite3.Connection) -> ActivationState | None:
    row = conn.execute(
        "SELECT * FROM activation_state WHERE id = 1"
    ).fetchone()
    if row is None:
        return None
    payload = {k: row[k] for k in _STATE_COLUMNS}
    payload["rng_state"] = json.loads(row["rng_state"])
    payload["warmup_complete"] = bool(row["warmup_complete"])
    return ActivationState.from_json(payload)


def _state_row(state: ActivationState) -> tuple:
    j = state.to_json()
    return (
        j["minute"],
        j["drive"],
        j["tone"],
        j["drift"],
        j["hazard"],
        j["theta"],
        j["cycle_id"],
        j["cycle_started_minute"],
        j["cycle_seq"],
        json.dumps(j["rng_state"], separators=(",", ":")),
        j["state_version"],
        j["policy_version"],
        j["warmup_started_minute"],
        1 if j["warmup_complete"] else 0,
        j["runs_total"],
        _now_iso(),
    )


def insert_state(conn: sqlite3.Connection, state: ActivationState) -> None:
    """首次初始化。已存在则抛 IntegrityError（这是正确行为：绝不重新初始化）。"""
    conn.execute(
        "INSERT INTO activation_state (id, "
        + ", ".join(_STATE_COLUMNS)
        + ", updated_at) VALUES (1, "
        + ", ".join("?" * (len(_STATE_COLUMNS) + 1))
        + ")",
        _state_row(state),
    )


def save_state(
    conn: sqlite3.Connection, state: ActivationState, expected_version: int
) -> None:
    """乐观锁写回。expected_version 是本次 tick 读到的版本号。"""
    assignments = ", ".join(f"{c} = ?" for c in _STATE_COLUMNS)
    cur = conn.execute(
        f"UPDATE activation_state SET {assignments}, updated_at = ? "
        "WHERE id = 1 AND state_version = ?",
        _state_row(state) + (expected_version,),
    )
    if cur.rowcount != 1:
        raise StateConflict(
            f"state_version 对不上（期望 {expected_version}），本次 tick 回滚"
        )


# --------------------------------------------------------------------------
# 机会 / 事件 / 快照
# --------------------------------------------------------------------------

# 机会失效时长。这是 Dispatcher 层的关切（“这次机会还新鲜吗”），不是策略参数，
# 所以不放进 WakePolicy——否则会白白改变 policy fingerprint。
OPPORTUNITY_TTL_MINUTES = 10


def opportunity_id(minute: int, cycle_seq: int) -> str:
    """确定性 id：同一分钟同一 cycle 重跑 tick 只会得到同一个 id。

    配上 PRIMARY KEY，幂等就是数据库保证的，不靠应用层记得去查重。
    """
    return f"wk_{minute:09d}_{cycle_seq:06d}"


def insert_opportunity(
    conn: sqlite3.Connection,
    opp_id: str,
    source: str,
    cycle_id: str,
    created_minute: int,
    detail: dict | None = None,
    priority: str = "normal",
    replay: str = "none",
    ttl_minutes: int = OPPORTUNITY_TTL_MINUTES,
) -> bool:
    """插入一个运行机会。已存在则返回 False（幂等，不报错）。"""
    cur = conn.execute(
        "INSERT OR IGNORE INTO wake_opportunities "
        "(id, source, cycle_id, created_minute, expires_minute, priority, replay, "
        " status, created_at, detail) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'CREATED', ?, ?)",
        (
            opp_id,
            source,
            cycle_id,
            created_minute,
            created_minute + ttl_minutes,
            priority,
            replay,
            _now_iso(),
            json.dumps(detail or {}, ensure_ascii=False, separators=(",", ":")),
        ),
    )
    return cur.rowcount == 1


def insert_event(
    conn: sqlite3.Connection,
    kind: str,
    minute: int,
    cycle_id: str,
    detail: dict | None = None,
) -> None:
    conn.execute(
        "INSERT INTO wake_events (kind, minute, cycle_id, detail, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            kind,
            minute,
            cycle_id,
            json.dumps(detail or {}, ensure_ascii=False, separators=(",", ":")),
            _now_iso(),
        ),
    )


def insert_wake_event(conn: sqlite3.Connection, event: WakeEvent) -> None:
    insert_event(conn, event.kind, event.minute, event.cycle_id, event.detail)


def insert_snapshot(
    conn: sqlite3.Connection,
    state: ActivationState,
    lambda_per_hour: float,
    mmod: float,
    wake: bool,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO state_snapshots "
        "(minute, drive, tone, drift, lambda_per_hour, hazard, theta, mmod, "
        " cycle_id, wake) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            state.minute,
            state.drive,
            state.tone,
            state.drift,
            lambda_per_hour,
            state.hazard,
            state.theta,
            mmod,
            state.cycle_id,
            1 if wake else 0,
        ),
    )


def prune_snapshots(conn: sqlite3.Connection, before_minute: int) -> int:
    cur = conn.execute(
        "DELETE FROM state_snapshots WHERE minute < ?", (before_minute,)
    )
    return cur.rowcount


def expire_stale_opportunities(conn: sqlite3.Connection, now_minute: int) -> int:
    """过期未处理的机会直接丢弃，不补发。"""
    cur = conn.execute(
        "UPDATE wake_opportunities SET status = 'EXPIRED' "
        "WHERE status = 'CREATED' AND expires_minute < ?",
        (now_minute,),
    )
    return cur.rowcount


# --------------------------------------------------------------------------
# 杂项
# --------------------------------------------------------------------------

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def minute_to_iso(minute: int) -> str:
    return time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(minute_to_epoch_seconds(minute))
    )


def counts(conn: sqlite3.Connection) -> dict:
    def one(sql: str, *args) -> int:
        row = conn.execute(sql, args).fetchone()
        return int(row[0]) if row else 0

    return {
        "opportunities": one("SELECT COUNT(*) FROM wake_opportunities"),
        "opportunities_created": one(
            "SELECT COUNT(*) FROM wake_opportunities WHERE status = 'CREATED'"
        ),
        "opportunities_expired": one(
            "SELECT COUNT(*) FROM wake_opportunities WHERE status = 'EXPIRED'"
        ),
        "snapshots": one("SELECT COUNT(*) FROM state_snapshots"),
        "events": one("SELECT COUNT(*) FROM wake_events"),
        "suppressed": one(
            "SELECT COUNT(*) FROM wake_events WHERE kind = 'suppressed_spontaneous'"
        ),
    }
