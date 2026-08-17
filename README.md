# kli-wake-runtime

让 Agent 在没有任何外部事件时也能自然获得运行机会——不是闹钟，不是固定推送。

当前仓库只有**离线核心**三件：不可变策略、连续状态演化引擎、模拟器。
没有数据库、没有 HTTP、没有鉴权、没有 Agent。跑坏了没有代价，先把参数调顺再谈接入。

```
wake/policy.py     冻结的参数对象 + policyVersion + 指纹
wake/engine.py     纯函数：状态演化、λ(t)、hazard 累积、恢复语义
wake/simulate.py   离线跑几十天，看密度、间隔、burst、感知密度
tests/test_replay.py  确定性重放测试（比公式单测重要得多）
```

无第三方依赖，Python 3.10+。

## 跑起来

```bash
python tests/test_replay.py

python -m wake.simulate --days 30 --seed 7
python -m wake.simulate --days 30 --user-turns-per-day 12 --silent-rate 0.7
python -m wake.simulate --days 30 --csv /tmp/wake.csv
python -m wake.simulate --days 14 --sweep lambda_base_per_hour=1.2,1.5,1.8
```

输出里有两组数字，含义完全不同：

- **运行机会密度**：引擎给了多少次 wake。高不一定吵。
- **感知密度**：叠加假设 silent rate 和表达预算之后，用户实际会看到几条。这一组才是体验。

## 模型

三个内部状态，时间尺度刻意错开：

| 状态 | 均值 | 时间尺度 | 作用 |
|---|---|---|---|
| `activationDrive` D | 0.50 | τ=12min | 短期激活驱动力，每次真实 Agent Run 后 −0.10 |
| `latentActivityTone` T | 0.50 | τ=6h | 「这一阵整体偏活跃还是偏安静」 |
| `stochasticDriftState` X | 0.00 | τ=25min | 有惯性的短期随机波动 |

```
λ(t) = clamp( λθ · exp[ βD(D−μD) + βT(T−μT) + βX·X ] · Mmod , λmin , λmax )
H(t) = ∫ λ(s) ds        Θ = −ln(U) ~ Exp(1)，每 cycle 抽一次
H ≥ Θ  →  一次 SpontaneousWakeOpportunity，然后 H 归零、Θ 重抽
```

λ 不是概率，也不是「发消息概率」。它只决定获得一次运行机会的瞬时倾向。

关于一个容易混淆的点：「30 分钟内至少醒一次」的累计概率会随时间上升，但 λ(t) 不会因为
用户离开更久就机械升高。用户离线时长本身不被解释成担心或想念。如果「很久没回复」在某个
上下文里真的产生了主观意义，那应该由上游（EB / 记忆）产生调制贡献，再作用于 λ。

## 四个钉死的语义

这四条是从设计评审里拎出来的，不写清楚每一条都会变成隐蔽 bug。

**1. 时间网格固定。** 所有评估时刻量化到 UTC epoch 整分钟，dt 恒为 60s。不允许用「实际经过
了多少秒」当 dt——否则连续运行和重启续跑的步长边界不同，轨迹必然分叉，确定性重放就是假的。
不足一分钟的残余留给下一 tick，不插值。

RNG 保存**完整状态**而非 seed，每步消耗顺序固定：先 X，再 T，D 不消耗，换 cycle 时追加
一次 Θ。改这个顺序会让所有历史轨迹不可重放。

为什么必须数值积分而不是解析积分：λ 依赖含噪声的 D/T/X，H=∫λ 没有闭式解。60s 步长相对
分钟级的状态尺度足够精确；1s 步长纯属浪费——X 的尺度是 25 分钟。

**2. 长间隔恢复是三件独立的事。** 空档超过 15 分钟时：

- 状态**照常**按整分钟演化（D 该回归就回归，T/X 该漂就漂，RNG 照常消耗）
- 这段时间的 hazard 增量**丢弃**
- 当前 cycle **作废**，Θ 重抽，H 归零，记一条 `suppressed_spontaneous`

自发 Wake 是机会，不是欠账，绝不补发。最常见的实现错误是「直接重锚」——把停机 8 小时前的
D 原样冻回来。状态演化和 hazard 累积必须分开处理。

**3. Direct Wake 不干扰自发节律。** 精确事件（Calendar / MCP / iOS / OpenLoop 到期）只做
两件事：注入一个 WakeOpportunity，运行结束后 kick D。**禁止**触碰 `cycle.hazard` 和
`cycle.theta`。Direct Wake 很频繁导致 D 持续偏低、自发唤醒变少，这是预期行为不是 bug。

`report_agent_run()` 也不消耗 RNG，所以不会让轨迹分叉。它必须挂在 AgentRuntime 的统一出口，
覆盖 UserTurn / Direct / Spontaneous 全部三种真实运行——漏掉 UserTurn 是最容易犯的。

**4. 表达预算是硬闸。** `visibility=external` 受滚动窗口限制：每小时 2 条、每 24 小时 8 条
（warm-up 期间 3 条）。超限时 AgencyDecision 的 outcome 被强制改写为 silent，记
`silent_budget_exhausted`，同时把模型原本想说的内容存进日志。`act` / `internal` 不受限。

这是唯一一个不依赖模型自律的兜底。evidence 和 value 都是模型自己填的布尔，confidence 又
明确不设阈值，所以 silent 本来完全押在 prompt 纪律上——模型想全填 true 谁也拦不住。
注意这和「用最小间隔修 wake 密度」不是一回事：闸门作用于注意力开销，与 λ 完全解耦，
不扭曲随机质感。真正会让人后悔的不是它太吵，是它某天连着说了二十句而系统觉得一切正常。

## 调参

**太吵**：先降 `lambda_base_per_hour`（1.5→1.2），最干净，不改随机质感。刚跑完就容易再醒 →
增大 `k_run` 或延长 `d_tau_minutes`。只是不喜欢极端 burst → 降 `lambda_max_per_hour`
（8→6），别降整体基线。

**太安静**：先升 `lambda_base_per_hour`（1.5→1.8）。想要「某一阵突然话多」→ 提高 `x_sigma`
或 `beta_x`。想让「今天整体更活跃/安静」更明显 → 提高 `t_sigma` 或 `beta_t`。

**别做**：不要同时大幅提高 λθ + β + σ + λmax，会把整体频率和 burst 强度一起放大。不要用
固定最小间隔修太吵。不要靠提高 μD/μT 模拟「更黏人」，那会永久偏移基线。不要把 Θ 的分布
当日常旋钮。

改参数会改变随机轨迹，所以 `with_overrides()` 会自动给 version 加指纹后缀，避免模拟结果
被误标成基线 policyVersion。

## 架构边界（后续阶段的护栏）

```
Wake ≠ Agent            引擎只给运行机会，不决定行为
Spontaneous ≠ Event     内生随机与外部精确事件是不同来源
Modulation ≠ Decision   EB / 记忆 / 未闭环只能调制 λ，不能 wake_now
Opportunity ≠ Message   拿到机会不等于必须说话
Wake ≠ Expression       「被唤醒」本身不能成为表达理由
Expression ≠ Generation 先决定是否表达，再生成内容
Uncertainty → Silent    不确定时默认沉默
```

真正的产品风险不在 Wake 层，而在下游：整套设计的意义全押在「silent 是正常且高质量的终态」
上，可模型往往把「被唤醒」本身当成该说话的隐含指令，十次醒来九次开口。那时候 λ(t) 再优雅，
体验上还是个漂亮的推送定时器。

所以 λ、D/T/X、Θ、Mmod、EB 调制分数、甚至 `source="spontaneous"` 都不进 Agent 上下文——
模型看到「系统这么高概率叫醒我」就会反推「所以我应该很想她」。`diagnostics()` 只写日志。

## 路线

- [x] 阶段 1：policy / engine / simulate + 重放测试（离线，无副作用）
- [ ] 阶段 1.5：离线跑够久，参数调到符合直觉，再往下走
- [ ] 阶段 2: 持久化（SQLite，BEGIN IMMEDIATE + stateVersion 乐观锁）、Supervisor reconcile
- [ ] 阶段 3: AgencyDecision 协议 + Agency Gate，验证 silent 能力
- [ ] 阶段 4: 统一鉴权 + SourceContract，然后才接第一个外部 Source

阶段 4 之前不要接外部源：那等于开一堆能远程叫醒你 Agent 的 HTTP 入口。

V1 不做 lease 续租、dispatcher 队列、attention_cost 加权。单用户单 Agent，一把互斥锁 +
SQLite 事务就够，不要为了架构完整度提前写。

参数是开发校准基线，不是心理学常量。允许调，但不改 Wake / Event / Modulation / AgentAgency
的结构边界。
