# 审批状态机(第 11 步)

审批门(`agent/approval.py`)对每一次工具调用做出放行/阻塞决定。它**只**依据工具声明的 `RiskTier`,从不解析自由格式字符串,因此 LLM 无法用措辞绕过边界。

## 状态

`ApprovalState`:`proposed -> validated -> awaiting_approval -> approved -> executed`,以及 `rejected` / `blocked`。

- `proposed`:计划中的调用。
- `validated`:参数已通过类型化 `StrictModel` 校验。
- `awaiting_approval`:`cluster_plan` 调用缺少显式人工批准,运行在此处停下。
- `approved`:审批门放行。
- `executed`:确定性函数成功运行(记录在 `ToolResult.approval_state`)。
- `rejected`:参数校验失败。
- `blocked`:被策略或风险层拒绝。

Agent 自身只能把调用推进到 `validated`;跨入 `cluster_plan` / `cluster_execute` 必须有独立的人工步骤。

## 策略(`ApprovalPolicy`)

```json
{
  "allow_local_writes": true,
  "approved_cluster_plan_call_ids": ["T5"],
  "approver": "researcher"
}
```

- `allow_local_writes`(默认 `true`):是否允许 `local_write` 工具写盘。设为 `false` 时,所有本地写工具都被 `blocked`。
- `approved_cluster_plan_call_ids`:被人工显式批准的 `cluster_plan` 调用 `call_id` 列表。
- `approver`:批准人姓名;批准 `cluster_plan` 时**必须**提供,否则即使 `call_id` 在批准列表中也会 `blocked`。

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
| `cluster_execute` | 任何情况 | `blocked` |

## runner 行为

`agent/runner.py` 逐个调用:注册表查找 -> 参数校验 -> 审批决策 -> 执行确定性函数。任何非 `ok` 结果(`blocked`/`validation_error`/`error`)都会让运行 `halted`,因为后续步骤通常依赖前一步。异常映射:

- 参数 `ValidationError` -> `validation_error`(`approval_state=rejected`)。
- 确定性校验器 `ValueError` -> `validation_error`。
- `OSError`/`KeyError`/`TypeError`/`RuntimeError` -> `error`。
- 未知工具 -> `blocked`,消息标注 "unknown tool"。

`AgentTranscript` 始终固定 `cluster_execution="disabled"`、`human_review_required=True`;`cluster_execute` 层的结果必须为 `blocked`,由模型校验器强制。

## LangGraph 人在环(HITL)

可选的 LangGraph 绑定(`agent/graph.py`)把上表映射为一张图:`cluster_plan` 调用在执行前调用 LangGraph 的 `interrupt`,图在 checkpointer 上暂停,`run()` 返回的 `pending_approvals` 列出待批准调用。人工用 `VaspAgentGraph.resume(result, {call_id: {"approve": true, "approver": "name"}})` 恢复;恢复后审批门用带批准的策略重新评估该调用。Agent 不能批准自己的 `cluster_plan` 调用——批准只能来自 `resume` 的显式人工输入或 `--policy` 预批准。批准动作在任何 `evaluate_call` 副作用之前收集,因此恢复时节点重跑不会重复执行前面的工具。
