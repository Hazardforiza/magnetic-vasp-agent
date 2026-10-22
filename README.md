# Magnetic VASP Agent

面向磁性材料 VASP 计算的本地控制面。项目把“提出计算方案”和“执行计算”分开：

- 确定性代码解析 POSCAR/mCIF、生成并校验 INCAR/KPOINTS、选择本机授权的 POTCAR，并为每个文件写入哈希。
- Agent 只能调用注册表中的类型化工具，不能执行任意 shell、SSH、MPI 或 VASP 命令。
- 远程计算只支持显式配置的单节点路径，并经过输入审核、资源批准、只读 probe、MPI smoke、传输哈希和本次启动批准。
- 后处理、知识检索、磁性分析和网页工作台都输出可审查结果；“命令成功”不等于科学结论成立。

当前保留的代码基线是 2026-10-20 的完整功能快照，包含网页、VASPKIT、磁性分析、经验库和项目资源管理。发布仓库只保留示例站点配置；真实账号、端点、POTCAR、API key 和运行产物必须留在使用者本机。

## 功能

| 模块 | 能力 |
| --- | --- |
| 输入内核 | POSCAR、mCIF、POTCAR 注册表、类型化 INCAR/KPOINTS、跨文件校验 |
| 工作流 | SCF、relax、DOS、Band、SOC/MAE、参数扫描、依赖和 manifest |
| Agent | LangGraph 可选绑定、只读/本地写入/集群计划/集群执行风险分级、人工审批 |
| 执行 | 单节点 Intel MPI 的 probe、资源计划、smoke、受控传输、启动、状态和回传 |
| 后处理 | OSZICAR、OUTCAR、vasprun.xml 解析；局域磁矩、能量、费米能、DOS 摘要和交叉核对 |
| VASPKIT | 固定任务 303/211/111、受限数字任务、DOS/能带绘图；支持 native/WSL |
| 磁性分析 | 磁构型能量比较、MAE、Heisenberg 拟合、U/应变敏感性 |
| 知识与协作 | 受治理来源清单、BM25/Qdrant 检索、引用草稿、人工审核记录、经验库和项目资源 |
| Web | 仅绑定 `127.0.0.1` 的本机工作台：输入生成、人工审核、Agent 会话、节点只读观察、结果导入 |

## 安装

要求 Python 3.11（不高于 3.12）和 `uv`。在仓库根目录执行：

```powershell
uv sync --all-extras --frozen
uv run --frozen magvasp doctor
```

只使用部分功能时可安装：

```powershell
uv sync --extra agent --extra web --frozen
uv sync --extra dev --frozen
```

## 最小使用流程

### 1. 注册本机 POTCAR

POTCAR 受许可证保护，不进入 Git、知识库或 LLM 提示。每台电脑独立扫描：

```powershell
uv run --frozen magvasp potcar scan `
  --root "C:\path\to\potpaw_PBE.54" `
  --family PBE_54 `
  --output var\potcar_registry.json
```

### 2. 配置模型（可选）

```powershell
Copy-Item config\agent\llm.example.json config\agent\llm.json
$env:MAGVASP_LLM_API_KEY = "在当前终端设置，不要写入 JSON"
```

`config/agent/llm.json` 只保存 provider、模型、URL 和环境变量名，不保存密钥。

### 3. 配置个人集群身份（可选）

`config/execution/site.team.json` 是占位模板，不能直接连接真实集群。复制个人模板并填写自己的账号：

```powershell
Copy-Item config\execution\identity.example.json config\execution\identity.json
# 编辑 user、connection；execution_enabled 默认保持 false
uv run --frozen magvasp execution compose `
  --site config\execution\site.team.json `
  --identity config\execution\identity.json `
  --output config\execution\cluster.json
```

`identity.json`、生成的 `cluster.json`、`.env` 和 `var/` 都不应提交。换用其他集群时，可直接参考 `config/execution/cluster.example.json`。

### 4. 启动本机网页

```powershell
uv run --extra agent --extra web --frozen magvasp web `
  --profile config\execution\cluster.json
```

默认地址是 `http://127.0.0.1:8765`。端口占用时加 `--port 8766`；不自动打开浏览器时加 `--no-browser`。也可以运行根目录的 `start-web.bat`。

## 命令示例

### 生成和审核输入

```powershell
uv run --frozen magvasp structure show tests\fixtures\Mn3Al\POSCAR

uv run --frozen magvasp potcar select `
  --registry var\potcar_registry.json `
  --policy config\potential_policy.json `
  --poscar tests\fixtures\Mn3Al\POSCAR `
  --candidate-id Mn3Al `
  --task static `
  --magnetic-mode collinear `
  --output var\Mn3Al.potential-plan.json

uv run --frozen magvasp workflow from-mcif `
  --mcif tests\fixtures\generated_mcif\candidate_01_FeO_BNS2_7.mcif `
  --output var\feo.request.json `
  --task static `
  --include-dos

uv run --frozen magvasp workflow build `
  --request var\feo.request.json `
  --registry var\potcar_registry.json `
  --policy config\potential_policy.json `
  --output var\bundle

uv run --frozen magvasp workflow review-card --bundle var\bundle
uv run --frozen magvasp workflow review `
  --bundle var\bundle `
  --reviewer alice `
  --decision accept
```

`workflow build` 只写本地输入包，绝不运行 VASP。接受审核只表示这版输入经过具名人工检查，不是计算授权。

### 研究计划与结果

```powershell
uv run --frozen magvasp research from-mcif `
  --mcif tests\fixtures\generated_mcif\candidate_01_FeO_BNS2_7.mcif `
  --output var\research-plan `
  --task magnetic_order --task relax --task static --task dos `
  --task mae --task convergence

uv run --frozen magvasp workflow build `
  --request var\research-plan\workflow.request.json `
  --registry var\potcar_registry.json `
  --policy config\potential_policy.json `
  --output var\research-bundle

uv run --frozen magvasp postprocess show path\to\calculation
uv run --frozen magvasp postprocess parse path\to\calculation `
  --output var\calculation.result.json
uv run --frozen magvasp postprocess plot-dos path\to\TDOS.dat `
  --output var\tdos.png
uv run --frozen magvasp postprocess plot-band path\to\REFORMATTED_BAND.dat `
  --labels path\to\KLABELS `
  --output var\band.png
```

结果解析是只读的。`converged`、输入审核或能量排名都不能单独证明基态、能隙、MAE 或其他科学结论。

### 磁性分析

```powershell
uv run --frozen magvasp magnetic-analysis compare results.json
uv run --frozen magvasp magnetic-analysis mae mae-results.json
uv run --frozen magvasp magnetic-analysis exchange exchange-results.json
uv run --frozen magvasp magnetic-analysis sensitivity results.json --parameter U
```

输入记录必须由用户保证同一物理比较条件。分析器会保留兼容性和科学验证边界，不会替用户宣布结论。

### Agent 与知识库

```powershell
uv run --frozen magvasp agent tools
uv run --frozen magvasp agent run-plan --plan plan.json
uv run --extra agent --frozen magvasp agent chat `
  --goal "读取一个 mCIF 并生成待审核的静态 SCF 输入" `
  --config config\agent\llm.json `
  --output var\agent.json

uv run --frozen magvasp knowledge inventory `
  --catalog config\knowledge\sources.json `
  --output var\knowledge\inventory.json
uv run --frozen magvasp skill list
```

知识检索只产生带来源的证据或 `draft_only` 草稿。技能文件只读，脚本永不执行。

## 远程执行闸门

远程执行默认关闭，且不是“配置一个布尔值就能提交”。完整路径必须有：

1. 工作流和输入 manifest 通过校验；
2. 具名人工审核；
3. 目标节点在站点白名单内，跳板机不能作为计算节点；
4. 最新只读 probe 和人工资源授权；
5. 固定 MPI smoke 通过；
6. 新的唯一远端目录完成逐文件 SHA256 核验；
7. 对本次 launch 的再次具名批准和精确确认词。

命令默认只生成预览：

```powershell
uv run --frozen magvasp execution plan --bundle var\bundle --profile config\execution\cluster.json --request request.json
uv run --frozen magvasp execution probe --profile config\execution\cluster.json --node node01
uv run --frozen magvasp execution transfer --profile config\execution\cluster.json --bundle var\bundle --request request.json
```

项目不提供任意 shell、跨节点 MPI、自动抢占、自动重试或取消他人作业。密码只能在人工 SSH 提示中输入，永不写入参数、环境变量、日志或仓库。

## 文档入口

- [用户配置与安全守则](USER_GUIDE.md)：每台电脑的安装、POTCAR、模型、集群身份和执行前检查。
- [部署指南](DEPLOYMENT.md)：新机器部署与排错。
- [网页说明](docs/web.md)：本机 Web 工作台、知识库和节点观察。
- [执行配置契约](docs/execution/cluster_profile.md)：站点模板、个人身份和 SSH 边界。
- [执行设计](docs/execution/real_cluster_execution_design.md)：远程执行的阶段、证据和人工闸门。
- [后处理](docs/postprocess.md)：VASP 输出和 VASPKIT 文件的解析边界。
- [研究工作流](docs/research_workflows.md)：mCIF-first 研究计划和科学限制。
- [知识治理](docs/knowledge/governance.md)：来源、引用、冲突和人工审核。

## 项目结构

```text
src/magnetic_vasp_agent/   确定性内核、CLI、Web、Agent 工具
config/                    示例策略、请求、工作流和知识清单
docs/                      运行、治理和科学边界说明
tests/                     离线单元测试与紧凑 fixture
scripts/                   只读审计/参考脚本；不作为 Agent 任意命令入口
var/                       本机生成物，Git 忽略
```

## 测试与分支策略

```powershell
uv run --frozen pytest
uv run --frozen ruff check .
```

测试使用离线 fixture、mock SSH/SFTP 和合成 POTCAR，不连接真实集群，也不上传授权数据。仓库只保留一个默认分支 `main`；临时开发分支完成合并后删除，避免把不同阶段的代码误当成并行版本。

## 安全声明

不要提交以下内容：`.env`、API key、私钥、`config/agent/llm.json`、`config/execution/identity.json`、生成的 `cluster.json`、POTCAR 正文、VASP 原始输出、`var/`、原始课程/文献资料和第三方 VASPKIT 发行包。若历史提交曾包含这些内容，应在 GitHub 上撤销相关凭据并按组织流程清理历史。
