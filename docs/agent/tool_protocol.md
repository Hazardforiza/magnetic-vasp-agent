# Agent 工具协议(第 11 步及受控启动扩展)

本文档描述把确定性 VASP 内核暴露给 LLM Agent 的类型化工具协议。协议层与审批状态机、runner 都不依赖任何 LLM SDK 或网络,可完全离线测试;真实 LLM 路径由可选的 LangGraph 绑定提供。

## 分层结构

```
agent/tool_models.py   协议与溯源类型(StrictModel + 规范 JSON SHA256)
agent/tools.py         工具注册表:把现有确定性函数包成类型化工具
agent/approval.py      分级审批状态机
agent/runner.py        执行 AgentPlan -> AgentTranscript(无 LLM；集群动作仍受审批与运行时闸门约束)
agent/llm_config.py    OpenAI 兼容 LLM 配置(密钥只从环境变量读)
agent/graph.py         LangGraph 薄绑定(仅在安装 langgraph/langchain 时可用)
```

选用 LangGraph 的原因:审批状态机与图天然同构,`cluster_plan` / `cluster_execute` 的“等待人工批准”直接用 LangGraph 的 `interrupt` 原语表达,得到真正的人在环而不放松任何边界。`VaspAgentGraph` 持有带 checkpointer 的已编译图,因此暂停的批准能在同一检查点上 `resume`。

## 硬边界

- LLM 只能调用注册表里存在的工具;不接受任何自由格式的 shell / SSH / MPI 字符串。
- 工具参数必须先通过其类型化 `StrictModel` 校验,输出仍会经过现有确定性校验器(`incar`/`kpoints`/`workflow` 等)。
- 通用集群执行保持禁用：没有任意 shell、`cancel` 或 `retry` 工具。`cluster_execute` 仅注册固定参数的 `execution.smoke`、`execution.transfer` 和 `execution.launch`；三者必须分别经过具名人工批准，且仍受各自的确定性运行时闸门约束。`execution.launch` 还会重新核验最新探测、资源批准、smoke、传输报告和确认口令。完整 Agent 真实端到端链路尚未实测。
- POTCAR 内容不会出现在任何工具结果里,也不会发给 LLM;`potcar.build` 只返回 manifest(元数据 + 哈希)。
- API key 只从环境变量读取,绝不写入配置文件、日志或提交到 Git,也不发往集群。

## 风险分级与已注册工具

| 风险层 | 语义 | 工具 |
| --- | --- | --- |
| `read_only` | 只解析/校验/检索或读取本地状态，不写盘；`execution.probe` 只预览固定探测协议，不连接集群 | 结构/INCAR/KPOINTS/mCIF/POTCAR/workflow 查看与校验、`skill.list/show/read`、`knowledge.search/answer`、`execution.probe/plan/inspect_run/gate_inspect`、`postprocess.parse`、`postprocess.vaspkit_catalog` |
| `local_write` | 在调用方指定的输出路径生成本地文件，放行但记录 | POTCAR/INCAR/KPOINTS/workflow 生成、mCIF workflow、MAE scan、`execution.request_create`、`execution.transfer_request_create`、`execution.simulate`、`knowledge.evidence/draft`、`postprocess.vaspkit`、`postprocess.vaspkit_task`、`postprocess.plot_dos`、`postprocess.plot_band` |
| `cluster_plan` | 可能读取真实集群状态、回传受限结果，或生成资源/传输计划；需显式具名人工批准 | `execution.probe_live`、`execution.topss_snapshot`、`execution.resource_plan`、`execution.resource_grant`、`execution.transfer_plan`、`execution.remote_status`、`execution.fetch` |
| `cluster_execute` | 对集群产生受控副作用；每次调用需独立具名人工批准 | 固定的 `execution.smoke`、`execution.transfer`、`execution.launch`；各自还需运行时证据；`cancel` / `retry` 不注册 |

`execution.resource_grant` 只有在本次 `cluster_plan` 调用获具名批准、参数中的 `approver` 与实际批准人一致、确认词精确为 `I_APPROVE_RESOURCE_GRANT` 时才执行。它在批准后即时重探测选定节点，再用已接受输入和不超过 60 秒的快照签发最长 120 分钟的授权。`execution.smoke` 与 `execution.launch` 也各自在自己的批准通过后即时重探测；launch 使用 launch-time 快照，不要求早期规划快照跨越 smoke、传输及多轮人工审批。传输不把占用快照当作分配证明，仍受有效授权、不可覆盖目录和逐文件哈希核验约束。

## 数据流

```
自然语言目标
  -> LangGraph planner 节点(只能调用注册工具)
  -> 工具注册表(类型化 args 校验 + 现有确定性校验器)
  -> 审批状态机(按风险层放行/阻塞;cluster_plan / cluster_execute 用 interrupt 暂停等人工批准)
  -> 确定性内核函数
  -> AgentTranscript(可审计,带 sha256)
```

## 协议对象

- `ToolSpec`:工具的公开描述(name/description/risk_tier/produces_files/input_schema),可安全暴露给 LLM。
- `ToolCall`:一次调用(`call_id`、`tool`、类型化 `arguments`)。
- `ToolResult`:结果(status ∈ `ok`/`validation_error`/`blocked`/`error`、`approval_state`、`output`、`output_sha256`)。`output_sha256` 由规范 JSON 计算并在模型内自校验。
- `AgentPlan`:有序的 `ToolCall` 列表,`call_id` 唯一;`digest()` 给出计划哈希。
- `AgentTranscript`:一次运行的可审计记录，始终保持 `human_review_required=True`。固定集群工具只有在具名批准且成功执行时才能产生非阻塞的成功结果；此时 `cluster_execution="controlled"`，否则为 `disabled`。该字段表示发生过受控集群动作，不单独证明 VASP 已启动；`execution.launch` 仍必须绑定运行时记录和全部闸门证据。

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

`agent chat` 是一次性调用:若出现 `cluster_plan` 或 `cluster_execute` 需要人工批准,运行会暂停并在 `pending_approvals` 中列出待批准调用(退出码 1)。CLI 路径下可用 `--policy` 预批准(按风险层填入 `approved_cluster_plan_call_ids` 或 `approved_cluster_execute_call_ids`，并提供 `approver`);程序化的暂停/恢复用 `VaspAgentGraph.run()` 与 `.resume()`。

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
