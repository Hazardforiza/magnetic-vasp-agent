# 真实集群执行设计（第 13 步）

本文档定义在共享生产集群上执行一次 VASP 计算所需的流程、证据和人工闸门。前半部分是设计和审查基线；当前工作区另有固定的 `execution.launch` 运行时入口，只有在这些证据和人工批准齐全时才会尝试一次远程启动。它不提供任意 SSH、SFTP、MPI、取消或重试接口。完整的 Agent 真实端到端链路仍未实测。

## 适用边界

执行协议本身不绑定单一站点。以下值必须从部署提供的 `ClusterProfile` 读取：网关地址/主机名、用户名、端口、计算节点白名单、VASP 模块、MPI 路径和 VASP 可执行文件路径。本文中的站点名、用户和节点均只是部署示例，换集群时由新的 profile 替换，代码和 Agent 工具无需改写。

- SSH 入口、用户名和端口必须从当前操作者自己的 `ClusterProfile` 读取；本文只使用示例端点 `gateway.example.org:22` 或本机隧道 `127.0.0.1:2222`，登录后必须验证部署 profile 中的跳板主机名（示例为 `login01`）和当前 identity 用户。
- `login01` 只能作为跳板，禁止在其上运行 VASP、MPI 或其他计算任务。
- 示例计算节点只有 `node01`、`node02`。实际白名单必须由管理员写入本机站点模板；示例值、历史资料或负载快照都不能授权运行。
- 目标是获得组内批准的**单个节点**，在该节点上使用站点确认的 Intel MPI 和 VASP 版本。
- 节点负载、进程列表、CPU 拓扑和 `PSR` 只能作为观察结果，不能证明独占资源或授权分配。
- 默认/示例 profile 仍为 `execution_enabled=false`、`remote_root=null`；个人合成 profile 即使显式启用，也必须在目录归属、资源窗口和本次人工批准齐全后才能上传或启动。
- 不得访问白名单外节点，不得读取、修改或删除其他用户的目录和进程，不得发送信号、取消或重启他人作业。
- POTCAR 内容不得进入日志、知识库、LLM 提示、Git 或审计记录；只记录其 manifest 和哈希。

## 目标与非目标

目标是把一次执行拆成可复核的阶段，每一步都有明确输入、输出、审批人和失败状态；远程命令只能来自固定的、类型化的实现，不接受模型或用户提供的任意 shell 字符串。

本设计不定义队列系统、跨节点 MPI、自动资源抢占、自动重试、自动清理、科学结论判定或无人工参与的执行 API。当前代码仍只提供离线执行计划、固定只读 probe、受控传输实现和 `remote_mpi` 的 blocked 记录。

未来的真实执行边界应是独立的本地 executor/control plane：Agent 只生成引用授权工件的结构化 launch request，不持有 SSH 私钥，也不接收任意命令。executor 在消费请求时重新读取 profile、工作流包和所有证据，校验授权签名、profile 哈希、时间窗口、一次性 nonce、节点白名单、目录归属及输入/argv/environment 哈希，然后才构造固定 argv 并以 `shell=False` 启动。Agent 注册表永远不直接获得 SSH、MPI 或 VASP 启动能力。

## 阶段与不可跳过的闸门

| 阶段 | 必须保存的证据 | 允许的副作用 | 放行条件 |
| --- | --- | --- | --- |
| `draft` | workflow/step manifest、四个输入文件哈希、POTCAR manifest 哈希、VASP executable 和目标 step | 本地内存 | 输入包通过现有 workflow、INCAR、KPOINTS、POSCAR、POTCAR manifest 校验 |
| `locally_validated` | 规范化计划及其哈希、依赖和科学参数审核记录 | 写本地计划/报告 | 所有上游 artifact 已验证，输入未被重写 |
| `resource_approval_required` | 具名批准人、节点、时间窗口、MPI ranks、CPU ID、独占规则 | 无远端副作用 | 人工明确批准单节点资源；不能由 snapshot 自动推断 |
| `preflight_passed` | gateway/node 身份、白名单检查、最新 topology/affinity/process/load/memory、module 和可执行文件路径、时间戳与哈希 | 只读 SSH | 入口为部署 profile 中的跳板（示例为 `login01`），目标为白名单节点，用户和主机身份匹配 |
| `smoke_test_approval_required` | smoke test 命令摘要、CPU 绑定、预计时长和输出位置 | 无远端副作用 | 人工批准无科学结果的最小单节点 Intel MPI 测试 |
| `smoke_test_passed` | stdout/stderr、退出码、主机名、rank map、绑定信息、开始/结束时间 | 仅运行批准的 smoke test | 单节点、绑定符合批准 CPU、无跨节点 rank、退出码为 0；失败立即停止 |
| `transfer_approval_required` | 唯一远端叶目录、文件清单、逐文件本地哈希、目录归属和容量确认 | 无远端副作用 | 人工确认 `remote_root` 是已存在且组内批准的专用目录 |
| `transferred_verified` | 远端目录创建结果、SFTP 结果、远端逐文件 SHA256、传输报告哈希 | 创建唯一叶目录并上传批准文件 | 禁止覆盖/符号链接；远端哈希与本地清单完全一致 |
| `launch_approval_required` | 最新预检、smoke test 证据、传输报告、科学审核、运行窗口和回滚联系人 | 无远端副作用 | 人工再次批准本次**单次** VASP 启动；不得复用过期 snapshot |
| `running` | 固定 argv、环境、PID/运行身份、启动时间、run ID 和目录 | 启动一次 VASP | 仅在已批准节点和唯一目录执行；无自动 retry |
| `completed_unverified` | 退出码、结束时间、受限日志摘要、原始目录位置 | 只读采集 | 不把退出成功当成科学收敛 |
| `postprocess_review_required` | 结构化 OUTCAR/vasprun.xml/OSZICAR 摘要、输入/输出哈希、人工解释记录 | 只读解析 | 结果版本、输入哈希和收敛证据齐全；科学结论仍需人工复核 |

任意阶段的身份、目录、资源、哈希、权限或协议检查失败都进入 `blocked` 或 `failed_manual_review`，不得自动跳过、升级、重试或清理。

## 预检与资源证明

预检必须分两层执行：本机到部署 profile 中跳板（示例为 `login01`）的 gateway identity，再由跳板通过固定内部 SSH 到一个已批准的计算节点。两层都验证 `hostname -s` 和 `id -un`，并拒绝把跳板当作计算节点。SSH 必须启用 `StrictHostKeyChecking=yes`、禁用 agent forwarding、端口转发和 ControlMaster；密码只可由人工终端输入，不能写入参数、环境变量、文件或日志。

只读 probe 收集 CPU topology、进程观察、负载、可用内存、当前 affinity、模块状态、MPI 路径和三个固定 VASP executable 路径。snapshot 必须带时区时间戳和完整协议哈希；它是当时的观察，不是预约、独占或授权证明。启动前必须重新执行或确认仍在人工批准窗口内的最新预检。

资源批准记录至少包含：节点名、有效时间窗口、独占/共享规则、MPI rank 数、每 rank 的 CPU ID、内存上限、批准人和批准时间。CPU ID 必须与拓扑一致且一一对应；不能把 `lscpu`、`/proc`、`PSR` 或空闲负载解释为组内分配。

## MPI smoke test

smoke test 是放行 VASP 前的独立闸门，只验证 Intel MPI 的单节点启动和绑定，不读取或产生科学结果。它必须使用固定、无任意 shell 的最小程序，显式记录 hostname、rank 数、rank-to-CPU 映射、stdout/stderr 和退出码；不得写入其他用户目录或使用跨节点参数。

测试失败、超时、主机名不符、出现跨节点 rank、绑定不符合批准 CPU 或输出协议异常时，状态保持 blocked。不得自动重试或顺延到另一节点；需要新的人工批准和新的审计事件。

## 受控传输

传输只允许现有 transfer 模块清单中的 `POSCAR`、`POTCAR`、`INCAR`、`KPOINTS`、每步 `manifest.json` 和顶层 `workflow.manifest.json`。`OUTCAR`、`CHGCAR`、`WAVECAR`、`vasprun.xml`、`DOSCAR`、`PROCAR` 等不得上传。

远端目录必须是 `<remote_root>/<workflow_id>--<run_id>` 这一唯一叶目录。实现先验证父目录存在且不是符号链接，再以 `test ! -e`、`test ! -L` 和 `mkdir` 防止覆盖；失败目录保留供人工检查，不自动删除。SFTP 完成后固定执行 `sha256sum`，逐项与本地 inventory 比对；任何不匹配都阻断后续启动。

`remote_root` 不能由历史资料、目录探测或 LLM 猜测。它必须由用户确认是组内批准、归属于当前账号、容量和权限足够的专用路径。

## 单次 VASP 启动

启动前闸门必须同时看到：最新预检、通过的 smoke test、已核验的传输报告、当前用户和目录归属、科学参数/上游依赖审核，以及明确的运行窗口。任何证据过期或哈希变化都回到前一闸门。

`prepare_execution` 生成的路径和 argv 只是离线候选预览，不能作为授权或执行输入。实际叶目录必须直接取自已批准且已核验的传输报告（`<remote_root>/<workflow_id>--<run_id>`）；预览与传输报告不一致时必须阻断。

实际 launcher 只能接受类型化的 `ExecutionRequest` 和已批准的 manifest，生成固定 argv；不得拼接任意 shell。目标为一个白名单节点、一个远端叶目录和一个 VASP executable（`vasp_std`、`vasp_ncl` 或 `vasp_gam`）。环境至少固定为：

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
I_MPI_PIN=1
I_MPI_PIN_PROCESSOR_LIST=<approved CPU IDs>
```

MPI rank 数、CPU 绑定、VASP 版本和 INCAR 中 `NPAR`/`NCORE` 的一致性必须在人工审核中确认；执行器不擅自改写 INCAR，也不自动调整 rank 数。每个 run 只允许一次启动尝试，启动失败保留现场并记录退出码。

### 授权工件

启动器只接受由人工审批流程产生的、不可覆盖的授权工件，不接受自然语言确认或单个布尔开关。工件至少绑定以下字段：

```json
{
  "schema_version": 1,
  "authorization_id": "AUTH-...",
  "workflow_manifest_sha256": "<64 hex>",
  "step_manifest_sha256": "<64 hex>",
  "input_sha256": {"POSCAR": "<64 hex>", "POTCAR": "<64 hex>", "INCAR": "<64 hex>", "KPOINTS": "<64 hex>"},
  "node": "node01",
  "remote_directory": "/approved/root/workflow--run",
  "mpi_ranks": 4,
  "cpu_ids": [0, 1, 2, 3],
  "preflight_snapshot_sha256": "<64 hex>",
  "smoke_test_sha256": "<64 hex>",
  "transfer_report_sha256": "<64 hex>",
  "argv_sha256": "<64 hex>",
  "environment_sha256": "<64 hex>",
  "window_start": "2026-09-12T10:00:00+08:00",
  "window_end": "2026-09-12T12:00:00+08:00",
  "approver": "researcher",
  "approved_at": "2026-09-12T09:30:00+08:00",
  "one_shot_nonce": "<random opaque value>",
  "consumed_at": null
}
```

实现必须逐字段比对授权工件、当前请求、最新预检、smoke 记录和传输报告；任何缺失、过期、哈希变化、窗口外时间或 nonce 已消费都保持 `blocked`。授权工件由人工控制的审批流程写入 append-only 审计存储，Agent 不能创建、修改或消费它；启动成功前只允许原子地标记 nonce 已消费，不能通过复制工件获得第二次启动机会。

## 监控、取消与重试

监控接口只能读取当前用户、当前 run ID 和当前唯一运行目录中的受限状态，不能扫描或解析其他用户作业，不能发送信号。状态采集应记录 PID、命令标记、目录、主机、时间戳和读取结果哈希。

取消是独立的高风险操作，必须由人工单独批准，并同时验证 PID 属于当前用户、命令包含本次 run 标记、工作目录等于已批准叶目录、主机等于批准节点。任何验证失败都拒绝，不尝试“猜测” PID。

重试不是原目录上的重新启动。它必须生成新的 `run_id`、新的唯一叶目录、全新的资源和启动批准，并重新完成预检、smoke test（若窗口或节点变化）和传输哈希校验；禁止覆盖原目录或自动重试。

## 后处理与失败处理

完成后只收集结构化摘要。后处理必须绑定 VASP 版本、输入文件哈希和来源文件哈希，并明确收敛状态、能量、磁矩、费米能和告警来源。`completed_unverified` 不等于物理收敛；结果需要人工科学复核，不能由 Agent 自动宣布结论。

远端失败、超时、协议异常、哈希不匹配或目录权限错误时：

1. 保留唯一远端目录和本地审计记录，不自动删除。
2. 不自动取消、重启、重试或切换节点。
3. 记录失败阶段、固定错误码、退出码、时间戳、相关哈希和人工后续动作。
4. 只有人工确认目录和作业归属后，才允许执行单独批准的回滚或取消。

回滚只可停止本次已确认且归属于当前用户的作业；不得触碰其他 PID、共享目录或跳板上的任何计算进程。

## 审计记录与 Agent 边界

每个阶段应写入不可覆盖的 `ExecutionAuditEvent`，至少包括 `run_id`、workflow/step manifest 哈希、阶段、动作、调用者、审批人、节点、目录、时间戳、输入/输出哈希、结果码和 blocker。审计记录不得包含 POTCAR 内容、密码、完整 OUTCAR 或未经脱敏的共享文件内容。

Agent 只能调用注册表中的类型化工具：`execution.probe`、`execution.plan`、`execution.inspect_run`、离线 `execution.transfer_plan` 和受控 `execution.launch`。`cluster_plan` 与 `execution.launch` 都需要具名人工批准；`execution.launch` 还必须重新核验最新探测、资源批准、smoke、传输报告和精确确认口令。`cancel` / `retry` 以及任意其他 `cluster_execute` 工具永不注册。自然语言、LLM 输出或 RAG 证据不能修改白名单、选择未批准节点、生成 shell、证明独占资源或绕过任一闸门。

受控启动仍应由独立的人工控制面消费上述授权工件；当前 `execution.launch` 是唯一窄化入口，不等于向 Agent 开放通用 `cluster_execute` 能力。任何新的真实副作用接口都必须先增加独立风险评审、审计格式和离线拒绝测试，再单独启用配置开关。

## 验收条件（仍可完全离线测试）

- 状态机拒绝跳过 smoke test、传输核验或启动前人工闸门。
- 过期或哈希变化的 snapshot、manifest、transfer report 不能进入 launch gate。
- 跳板（示例为 `login01`）和白名单外节点不能通过目标校验。
- 未确认 `remote_root`、目录已存在、符号链接或远端哈希不匹配都会阻断。
- smoke test 失败/超时不能启动 VASP，且不会自动重试。
- retry 必须生成新的 `run_id` 和远端叶目录；原目录不可覆盖。
- Agent 工具清单最多只能出现固定的 `execution.launch` `cluster_execute` 入口；不得出现任意 shell、`cancel` 或 `retry` 工具。
- 既有测试只使用 offline fixture、mock SSH/SFTP 和固定输出协议，不连接真实集群、不启动 MPI/VASP；真实 Agent E2E 需另行记录，不能用这些测试冒充。
