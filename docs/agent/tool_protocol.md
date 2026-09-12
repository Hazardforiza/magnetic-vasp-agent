# Agent 工具协议(第 11 步)

本文档描述把确定性 VASP 内核暴露给 LLM Agent 的类型化工具协议。协议层与审批状态机、runner 都不依赖任何 LLM SDK 或网络,可完全离线测试;真实 LLM 路径由可选的 LangGraph 绑定提供。

## 分层结构

```
agent/tool_models.py   协议与溯源类型(StrictModel + 规范 JSON SHA256)
agent/tools.py         工具注册表:把现有确定性函数包成类型化工具
agent/approval.py      分级审批状态机
agent/runner.py        执行 AgentPlan -> AgentTranscript(无 LLM、无集群)
agent/llm_config.py    OpenAI 兼容 LLM 配置(密钥只从环境变量读)
agent/graph.py         LangGraph 薄绑定(仅在安装 langgraph/langchain 时可用)
```

选用 LangGraph 的原因:审批状态机与图天然同构,`cluster_plan` 的“等待人工批准”直接用 LangGraph 的 `interrupt` 原语表达,得到真正的人在环而不放松任何边界。`VaspAgentGraph` 持有带 checkpointer 的已编译图,因此暂停的 `cluster_plan` 批准能在同一检查点上 `resume`。

## 硬边界

- LLM 只能调用注册表里存在的工具;不接受任何自由格式的 shell / SSH / MPI 字符串。
- 工具参数必须先通过其类型化 `StrictModel` 校验,输出仍会经过现有确定性校验器(`incar`/`kpoints`/`workflow` 等)。
- 集群执行保持禁用:没有 launch/cancel/retry 工具,`cluster_execute` 风险层始终阻塞。
- POTCAR 内容不会出现在任何工具结果里,也不会发给 LLM;`potcar.build` 只返回 manifest(元数据 + 哈希)。
- API key 只从环境变量读取,绝不写入配置文件、日志或提交到 Git,也不发往集群。

## 风险分级与已注册工具

| 风险层 | 语义 | 工具 |
| --- | --- | --- |
| `read_only` | 只解析/校验/检索,不写盘、不连集群,自动放行 | `structure.show`、`incar.show`、`incar.validate`、`kpoints.show`、`kpoints.validate`、`mcif.show`、`potcar.select`、`workflow.validate`、`workflow.inspect`、`knowledge.search`、`knowledge.answer`、`execution.probe`、`execution.plan` |
| `local_write` | 在调用方指定的输出路径生成本地文件(dry-run),放行但记录 | `potcar.build`、`incar.build`、`kpoints.build`、`workflow.build`、`execution.simulate`、`knowledge.evidence`、`knowledge.draft` |
| `cluster_plan` | 仅离线传输计划(无 `--connect`),需显式人工批准 | `execution.transfer_plan` |
| `cluster_execute` | 真实提交/取消/重试 | 永不注册,始终硬阻塞 |

## 数据流

```
自然语言目标
  -> LangGraph planner 节点(只能调用注册工具)
  -> 工具注册表(类型化 args 校验 + 现有确定性校验器)
  -> 审批状态机(按风险层放行/阻塞;cluster_plan 用 interrupt 暂停等人工批准)
  -> 确定性内核函数
  -> AgentTranscript(可审计,带 sha256)
```

## 协议对象

- `ToolSpec`:工具的公开描述(name/description/risk_tier/produces_files/input_schema),可安全暴露给 LLM。
- `ToolCall`:一次调用(`call_id`、`tool`、类型化 `arguments`)。
- `ToolResult`:结果(status ∈ `ok`/`validation_error`/`blocked`/`error`、`approval_state`、`output`、`output_sha256`)。`output_sha256` 由规范 JSON 计算并在模型内自校验。
- `AgentPlan`:有序的 `ToolCall` 列表,`call_id` 唯一;`digest()` 给出计划哈希。
- `AgentTranscript`:一次运行的可审计记录。即使写了本地文件,transcript 仍固定 `cluster_execution="disabled"`、`human_review_required=True`;`cluster_execute` 结果必须为 `blocked`。

## CLI

```powershell
# 列出工具与风险分级(只读,无 LLM)
uv run --frozen magvasp agent tools

# 跑手写/类型化计划(无 LLM,完全确定性)
uv run --frozen magvasp agent run-plan --plan plan.json

# 真实 LangGraph 路径(需 agent extra + 网络 + API key)
$env:MAGVASP_LLM_API_KEY = "sk-..."
uv run --frozen magvasp agent chat `
  --goal "为 Mn3Al 规划一次静态 SCF 输入" `
  --config config\agent\llm.json
```

`agent chat` 是一次性调用:若出现 `cluster_plan` 需要人工批准,运行会暂停并在 `pending_approvals` 中列出待批准调用(退出码 1)。CLI 路径下可用 `--policy` 预批准(在策略里填入 `approved_cluster_plan_call_ids` 和 `approver`);程序化的暂停/恢复用 `VaspAgentGraph.run()` 与 `.resume()`。

`plan.json` 示例:

```json
{
  "schema_version": 1,
  "goal": "inspect Mn3Al inputs",
  "calls": [
    {"call_id": "T1", "tool": "structure.show",
     "arguments": {"poscar": "tests/fixtures/Mn3Al/POSCAR"}},
    {"call_id": "T2", "tool": "kpoints.show",
     "arguments": {"kpoints": "tests/fixtures/Mn3Al/KPOINTS"}}
  ]
}
```
