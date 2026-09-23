# 审批状态机(第 11 步)

审批门(`agent/approval.py`)对每一次工具调用做出放行/阻塞决定。它**只**依据工具声明的 `RiskTier`,从不解析自由格式字符串,因此 LLM 无法用措辞绕过边界。

## 状态

`ApprovalState`:`proposed -> validated -> awaiting_approval -> approved -> executed`,以及 `rejected` / `blocked`。

- `proposed`:计划中的调用。
- `validated`:参数已通过类型化 `StrictModel` 校验。
- `awaiting_approval`:`cluster_plan` 或固定的 `cluster_execute` 调用缺少显式人工批准,运行在此处停下。
- `approved`:审批门放行。
- `executed`:确定性函数成功运行(记录在 `ToolResult.approval_state`)。
- `rejected`:参数校验失败。
- `blocked`:被策略或风险层拒绝。

Agent 自身只能把调用推进到 `validated`;跨入 `cluster_plan` / `cluster_execute` 必须有独立的人工步骤。`cluster_execute` 只允许固定的 `execution.smoke`、`execution.transfer` 和 `execution.launch`，每个调用都要单独、具名批准，不开放任意集群命令。审批只放行本次类型化调用；各工具仍执行自身的确定性运行时校验，其中 `execution.launch` 会重新核验最新探测、资源批准、smoke、传输哈希和确认口令。

## 策略(`ApprovalPolicy`)

```json
{
  "allow_local_writes": true,
  "approved_cluster_plan_call_ids": ["T5"],
  "approved_cluster_execute_call_ids": ["T9"],
  "approver": "researcher"
}
```

- `allow_local_writes`(默认 `true`):是否允许 `local_write` 工具写盘。设为 `false` 时,所有本地写工具都被 `blocked`。
- `approved_cluster_plan_call_ids`:被人工显式批准的 `cluster_plan` 调用 `call_id` 列表。
- `approved_cluster_execute_call_ids`:被人工显式批准的固定 `execution.smoke` / `execution.transfer` / `execution.launch` 调用 `call_id` 列表；批准一个调用不会批准另外两个，也不替代各工具的运行时证据校验。
- `approver`:批准人姓名;批准 `cluster_plan` 或 `cluster_execute` 时**必须**提供,否则即使 `call_id` 在批准列表中也会 `blocked`。

`execution.resource_grant` 还要求工具参数中的 `approver` 与本次 `cluster_plan` 的实际批准人完全一致，并要求精确确认词 `I_APPROVE_RESOURCE_GRANT`；Agent 不能用参数替人签名。该工具在批准后才执行固定只读重探测，因此 60 秒新鲜度从实际授权动作计算。`execution.smoke` 与 `execution.launch` 同样在各自独立批准后重探测，旧的资源参考快照不会被跨阶段复用。

策略由人控制,Agent 不能修改。

## 决策表

| 风险层 | 条件 | 结果 |
| --- | --- | --- |
| `read_only` | — | `approved`(自动) |
| `local_write` | `allow_local_writes=true` | `approved`(记录) |
| `local_write` | `allow_local_writes=false` | `blocked` |
| `cluster_plan` | `call_id` 未在批准列表 | `awaiting_approval` |
| `cluster_plan` | 已批准但无 `approver` | `blocked` |
| `cluster_plan` | 已批准且有 `approver` | `approved` |
| `cluster_execute` | 非 `execution.smoke` / `execution.transfer` / `execution.launch` 工具 | `blocked` |
| `cluster_execute` | 固定工具的本次 `call_id` 未获批准 | `awaiting_approval` |
| `cluster_execute` | 固定工具的本次 `call_id` 已批准且有具名 approver | `approved`，随后仍由工具核验运行时证据 |

## runner 行为

`agent/runner.py` 逐个调用:注册表查找 -> 参数校验 -> 审批决策 -> 执行确定性函数。任何非 `ok` 结果(`blocked`/`validation_error`/`error`)都会让运行 `halted`,因为后续步骤通常依赖前一步。异常映射:

- 参数 `ValidationError` -> `validation_error`(`approval_state=rejected`)。
- 确定性校验器 `ValueError` -> `validation_error`。
- `OSError`/`KeyError`/`TypeError`/`RuntimeError` -> `error`。
- 未知工具 -> `blocked`,消息标注 "unknown tool"。

`AgentTranscript` 始终要求 `human_review_required=True`。它只接受上述三个固定工具产生的 `cluster_execute` 结果：非阻塞结果必须绑定具名人工批准，成功结果还必须标为 `executed`。只要任一固定集群动作成功，摘要字段就是 `cluster_execution="controlled"`；否则为 `disabled`。`controlled` 只说明 transcript 中发生过受控集群动作，不等于 VASP 已成功启动或科学计算已完成；启动事实仍以 `execution.launch` 的运行时记录为准。

## LangGraph 人在环(HITL)

可选的 LangGraph 绑定(`agent/graph.py`)把审批状态机映射为一张图。自动 `interrupt` / `resume` 路径同时收集 `cluster_plan` 与 `cluster_execute` 批准，并在任何 `evaluate_call` 副作用之前一次性收集当前批次的高风险决定，避免恢复时重复执行已放行工具。Agent 不能批准自己的调用；预批准策略和交互式批准都不能替代运行时证据校验。该交互路径已有离线回归测试，但 Agent 独立完成一次真实集群端到端实验仍未实测。
