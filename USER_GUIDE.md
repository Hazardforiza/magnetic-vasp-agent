# 用户配置与安全守则

这份文件面向实际使用者。项目默认是本机、可审查、关闭远程执行的控制面；任何真实站点都必须由管理员和用户在本机配置。

## 一、必须遵守的规则

1. 每个人使用自己的 `identity.json`、API key、SSH key、POTCAR 和运行目录，不能复制他人的私有配置。
2. `execution_enabled` 初始必须是 `false`。输入审核通过不等于允许启动集群。
3. 密码只在人工 SSH 提示中输入；不能写进 JSON、`.env`、脚本参数、日志或聊天。
4. API key 只放环境变量（默认 `MAGVASP_LLM_API_KEY`），不能写入 `llm.json` 或 Git。
5. POTCAR 是本机授权数据，只记录来源元数据和哈希，不上传给 LLM、不提交 Git。
6. `remote_root` 必须包含当前用户自己的账号路径；不能使用他人的家目录。
7. 节点负载快照只表示当时观察，不表示资源已分配；启动前必须重新 probe 并取得具名批准。
8. 跳板机只能转发 SSH，不能在上面运行 VASP、MPI 或其他计算任务。
9. 所有输入包、传输报告和结果都要保留哈希；不要覆盖已有 workflow 或远端运行目录。
10. 不要把“脚本执行成功”“输入被接受”“电子步收敛”写成科学结论。

## 二、安装

要求 Python 3.11 和 `uv`：

```powershell
uv sync --all-extras --frozen
uv run --frozen magvasp doctor
```

只用网页和 Agent：

```powershell
uv sync --extra agent --extra web --frozen
```

## 三、配置本机 POTCAR

每台电脑单独扫描授权目录。扫描不会复制 POTCAR 正文：

```powershell
uv run --frozen magvasp potcar scan `
  --root "C:\path\to\potpaw_PBE.54" `
  --family PBE_54 `
  --output var\potcar_registry.json
```

检查 `var\potcar_registry.json` 是否只包含路径、元素、版本和 SHA256。不要把该文件或 POTCAR 目录发送给模型或上传到仓库。

## 四、配置模型

```powershell
Copy-Item config\agent\llm.example.json config\agent\llm.json
$env:MAGVASP_LLM_API_KEY = "your-key"
```

编辑 `config/agent/llm.json` 时只填写模型服务地址和模型名；`api_key_env` 指向环境变量名。换终端后需要重新设置环境变量。

验证：

```powershell
uv run --extra agent --frozen magvasp agent tools
```

`agent tools` 不调用模型，也不访问集群。第一次使用自然语言 Agent 时再运行：

```powershell
uv run --extra agent --frozen magvasp agent chat `
  --goal "读取一个 mCIF 并生成待审核的静态 SCF 输入" `
  --config config\agent\llm.json `
  --output var\agent.json
```

出现 `pending_approvals` 时必须由真实的人逐项批准；Agent 不能自批。

## 五、配置集群身份

### 5.1 站点模板

`config/execution/site.team.json` 只有示例值，如 `gateway.example.org`、`login01`、`node01` 和 `/opt/...`。管理员必须先在受控环境中替换为真实站点信息。模板可以共享，但不能直接连接真实集群。

### 5.2 个人身份

```powershell
Copy-Item config\execution\identity.example.json config\execution\identity.json
```

只编辑自己的字段：

```json
{
  "schema_version": 1,
  "user": "your_cluster_username",
  "connection": "office",
  "execution_enabled": false
}
```

`connection` 必须是模板中存在的 `office` 或 `tunnel`。如需覆盖端口或远端目录，只使用管理员批准的值。远端目录应为 `/home/<自己的账号>/magvasp/runs` 这一类专用路径。

### 5.3 合成本机 profile

```powershell
uv run --frozen magvasp execution compose `
  --site config\execution\site.team.json `
  --identity config\execution\identity.json `
  --output config\execution\cluster.json
```

`cluster.json` 是本机私有文件，默认被 Git 忽略。换机器时重新生成，不要复制旧文件。使用 `--overwrite` 前先确认目标路径是普通文件。

### 5.4 首次登录和只读 probe

首次验证可以让用户在终端输入密码：

```powershell
uv run --frozen magvasp execution login `
  --profile config\execution\cluster.json `
  --connect --interactive
```

只读检查：

```powershell
uv run --frozen magvasp execution probe `
  --profile config\execution\cluster.json `
  --node node01
uv run --frozen magvasp execution wannier-probe `
  --profile config\execution\cluster.json `
  --node node01
```

probe 不运行 MPI/VASP、不上传文件、不证明资源分配。后续无人值守访问需要操作系统提供的 SSH 公钥或 agent；程序不会接收密码参数。

## 六、生成、审核和执行输入

1. 用 `structure show`、mCIF 或网页导入结构。
2. 用 POTCAR registry、INCAR/KPOINTS recipe 生成工作流。
3. 用 `workflow inspect` 核对输入和 manifest 哈希。
4. 由具名真人运行 `workflow review-card` 并记录 `accept`、`revise` 或 `reject`。
5. 只有审核通过后，才可以生成执行计划；仍需新的资源批准、smoke、传输核验和 launch 批准。

```powershell
uv run --frozen magvasp workflow inspect var\bundle
uv run --frozen magvasp workflow review-card --bundle var\bundle
uv run --frozen magvasp workflow review `
  --bundle var\bundle `
  --reviewer your_name `
  --decision revise `
  --notes "补充 U/J 或收敛依据"
```

`reject` 和 `revise` 必须填写原因。审核经验写入本机 `var/reviews/`，推广为组内知识前必须再次人工审查。

## 七、网页工作台

```powershell
uv run --extra agent --extra web --frozen magvasp web `
  --profile config\execution\cluster.json
```

默认访问 `http://127.0.0.1:8765`。网页只监听回环地址，不是共享网站。它可以：

- 上传 mCIF/CIF，生成和审核输入；
- 运行带审批的本机 Agent 会话；
- 查看白名单节点的只读观察；
- 导入 OSZICAR、OUTCAR、vasprun.xml 和已生成的 DOS/能带文本；
- 浏览本机或已配置的共享知识库。

网页服务重启后，未完成的 Agent 会话不会自动恢复；项目、审核和结果保存于 `var/web/`。

## 八、VASPKIT 和后处理

Windows 上优先使用 WSL 后端，不要把 Linux ELF 当作 native Windows 程序：

```powershell
uv run --frozen magvasp postprocess vaspkit path\to\calc `
  --task tdos_dat --backend wsl --executable path\to\vaspkit
uv run --frozen magvasp postprocess plot-dos path\to\TDOS.dat `
  --output var\tdos.png
```

固定适配器只接受已实现的任务和格式。输出图可以帮助检查数据，不能替代对 VASP 输入、收敛和物理模型的人工复核。

## 九、常见错误

| 现象 | 处理 |
| --- | --- |
| `uv sync` 失败 | 确认 Python 3.11、`uv sync --all-extras --frozen` 和网络/缓存可用 |
| POTCAR 找不到 | 检查扫描根目录是否直接包含元素子目录，并重新生成 registry |
| Agent key 错误 | 检查当前终端的 `MAGVASP_LLM_API_KEY` 和 `llm.json` 的 `api_key_env` |
| profile 无法合成 | 检查 `connection`、用户名格式和 `remote_root` 是否属于当前用户 |
| 网页打不开 | 确认访问 `127.0.0.1`、端口未占用，必要时加 `--port 8766` |
| 远程 launch 被阻断 | 查看 blocker；不要绕过检查，重新完成对应人工批准和证据步骤 |
| 结果未收敛 | 保留原始目录和解析 JSON，不能自动重试或覆盖原 run |

## 十、提交前检查

```powershell
git status --short
git diff --check
git grep -n -I -E 'MAGVASP_LLM_API_KEY=|BEGIN (OPENSSH|RSA)|password=|/home/[A-Za-z0-9_.-]+/magvasp/runs'
uv run --frozen pytest
```

发现真实账号、端点、私钥、密码、API key、POTCAR 或运行产物时，先从提交中移除，并按组织流程撤销已暴露的凭据。
