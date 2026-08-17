"""Kli Wake Runtime —— 离线核心（policy / engine / simulate）。

这个包故意不包含：数据库、HTTP 路由、鉴权、Agent、AgencyGate、Dispatcher。
它只回答"此刻是否产生了一次运行机会"，其余都在后续阶段。
"""

from .engine import (
    ActivationState,
    WakeEvent,
    advance,
    diagnostics,
    epoch_minute,
    init_state,
    lambda_of_state,
    lambda_per_hour,
    report_agent_run,
    step,
)
from .policy import DEFAULT_POLICY, POLICY_VERSION, WakePolicy

__all__ = [
    "ActivationState",
    "WakeEvent",
    "WakePolicy",
    "DEFAULT_POLICY",
    "POLICY_VERSION",
    "advance",
    "diagnostics",
    "epoch_minute",
    "init_state",
    "lambda_of_state",
    "lambda_per_hour",
    "report_agent_run",
    "step",
]
