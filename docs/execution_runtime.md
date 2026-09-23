# Gated launch / status / cancel（第 13 步运行时）

本地模拟仍只跑固定进程。远程 MPI/VASP 只有在 `ClusterProfile.execution_enabled=true`、人工资源批准与最新探测相匹配、具名 smoke 报告通过、传输哈希核验完成，并且调用方给出精确口令 `I_CONFIRM_REMOTE_LAUNCH` 时才会启动。缺少任一证据时仍写 `blocked` 记录。Agent 注册表包含唯一的受控 `execution.launch` 入口；`cancel` / `retry` 不提供。完整 Agent 端到端真实链路尚未实测。

## 确认口令

布尔开关不够安全。调用方必须给出精确短语：

- 本地模拟：`I_CONFIRM_LOCAL_LAUNCH`
- 远程 MPI：`I_CONFIRM_REMOTE_LAUNCH`

短语不对则直接拒绝，不创建运行目录（远程在确认通过后才写 `blocked` 记录，便于审计「有人试图远程启动」）。

## 远程阻断原因

`remote_mpi` 记录固定带上：

- `REMOTE_EXECUTION_DISABLED`
- `SSH_KEY_LOGIN_UNVERIFIED`
- `MPI_SMOKE_TEST_NOT_PASSED`

缺少闸门时，`cluster_execution` 为 `"disabled"`；通过全部校验后，记录为 `"remote_mpi"` 并包含节点、远端目录、PID 和 VASP 可执行文件。`ClusterProfile.execution_enabled` 是 profile 字段，默认/示例值为 `false`，不是对任何 profile 的永久类型约束。

## 取消边界

`cancel` 只作用于：

- `backend=local_simulation`
- `owned_locally=true`
- 现场命令行仍包含标记 `SIMULATION ONLY`

否则拒绝。不会向集群发信号，也不会取消任意 PID。

## Agent

Agent 可调用固定参数的 `execution.launch`，但必须先经过工具审批，并提供 profile、workflow bundle、ExecutionRequest、资源批准、smoke 报告、传输报告和精确人工确认口令。运行时会再次校验这些文件的绑定关系、节点、核数和哈希；失败只生成 `blocked` 记录，不会降级为任意 SSH 或 shell。`execution.inspect_run` 仍是只读状态读取；没有 `cancel` / `retry` 工具。
