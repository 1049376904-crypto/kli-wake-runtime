# 部署（形态 B：无常驻进程）

真正的节律活在 SQLite 里，systemd timer 只是每分钟给它一次推进的机会。
漏跑、重启、手动 stop 全部走同一条恢复路径，而那条路径有测试覆盖。

## 先在前台跑几分钟

先不要装 systemd，手动敲几次看看：

```bash
cd ~/kli-wake-runtime
python3 -m wake.tick            # 首次会初始化
sleep 60 && python3 -m wake.tick
python3 -m wake.tick status
```

默认库在 `~/.local/share/kli-wake/kli_wake.db`，用 `KLI_WAKE_DB` 或 `--db` 改。

阶段 2 没有 Dispatcher 也没有 Agent，所以产生的机会会全部过期（status 里能看到
EXPIRED 计数在涨）。这是预期的——现在只是看节律在真实时间里长什么样。

## 装成 systemd timer

一步一步敲，不要整段粘贴。下面每一条都要确认没报错再跑下一条——
中间断掉一条（比如 mkdir）会让 timer 每分钟失败一次，而报错看起来与真正原因无关。

```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin kli
```
```bash
sudo mkdir -p /opt/kli-wake-runtime /var/lib/kli-wake
```
```bash
sudo cp -r wake /opt/kli-wake-runtime/
```
```bash
sudo chown -R kli:kli /var/lib/kli-wake
```

装单元之前先验一下两个目录都到位了：

```bash
ls -ld /var/lib/kli-wake && ls /opt/kli-wake-runtime/wake/
```

`/var/lib/kli-wake` 不存在时，`ReadWritePaths` 会让 systemd 连命名空间都搭不起来，
报 `status=226/NAMESPACE`。

```bash
sudo cp deploy/kli-wake-tick.service /etc/systemd/system/
```
```bash
sudo cp deploy/kli-wake-tick.timer /etc/systemd/system/
```
```bash
sudo systemctl daemon-reload
```
```bash
sudo systemctl enable --now kli-wake-tick.timer
```

看一眼：

```bash
systemctl list-timers kli-wake-tick.timer
journalctl -u kli-wake-tick.service -n 20 --no-pager
```

`journalctl` 里应该安静无错。看状态要带上库路径，并在 `/opt/kli-wake-runtime` 下跑：

```bash
cd /opt/kli-wake-runtime && \
  sudo -u kli KLI_WAKE_DB=/var/lib/kli-wake/kli_wake.db python3 -m wake.tick status
```

注意这个库和你在 `~` 下手敲时用的 `~/.local/share/kli-wake/kli_wake.db` 是两个，
会各自初始化一次。手敲的那个留着或删掉都无所谓。

`.service` 里开了 `PrivateNetwork=yes`。阶段 2 不需要网络，接外部 Source 之前别动这行。

`Persistent=false` 是故意的：停机期间错过的 tick 不补。自发 Wake 是机会，不是欠账。

## 看曲线

跑几天之后导出快照：

```bash
cd /opt/kli-wake-runtime && \
  sudo -u kli KLI_WAKE_DB=/var/lib/kli-wake/kli_wake.db \
  python3 -m wake.tick export --out /tmp/wake.csv --days 3
```

字段：minute, iso, drive, tone, drift, lambda_per_hour, hazard, theta, mmod, cycle_id, wake。
抱到 Excel / pandas 里画就行。重点看三个：真实时间里的 wake 密度跟离线模拟对不对得上
（离线是 32.8 次/天）、最长沉默有多长、T 会不会长时间贴边。

## 报告 Agent Run

任何真实运行结束后要 kick D，UserTurn 也算：

```bash
python3 -m wake.tick report-run --source user_turn
```

接进 dwell 之后这一步要挂在 AgentRuntime 的统一出口，不要让每个调用方自己记得触发。
漏掉 UserTurn 是最容易犯的——那会让 D 偏高，自发唤醒比设计频繁。

## 出问题时

`journalctl -u kli-wake-tick.service -n 20 --no-pager` 里的几种典型退码：

- `226/NAMESPACE` —— `/var/lib/kli-wake` 不存在。建目录。
- `2/INVALIDARGUMENT` —— 命令行参数不对。`ExecStart` 应该是 `-m wake.tick tick -q`。
- `ModuleNotFoundError: wake` —— `WorkingDirectory` 下没有 `wake/` 目录。

## 卸下来

```bash
sudo systemctl disable --now kli-wake-tick.timer
```

数据库留着。下次开回来会读旧状态继续跑，不会重新初始化——但中间那段空白会走
长间隔恢复：状态照常演化，hazard 丢弃，cycle 重抽。
