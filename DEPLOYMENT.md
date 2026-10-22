# Magnetic VASP Agent 部署指南

本文用于在一台新电脑上安装项目。仓库中的站点配置是占位模板；真实集群信息由管理员和每位用户在本机填写。

## 前提

- Windows 10/11、Linux 或 macOS；
- Python 3.11；
- `uv`；
- 本机获授权的 VASP POTCAR；
- 若使用 Agent：一个 OpenAI-compatible 模型服务；
- 若使用远程执行：管理员批准的 SSH 站点和个人账号。

## 安装

```powershell
uv sync --all-extras --frozen
uv run --frozen magvasp doctor
uv run --frozen pytest
```

## 本机资源

### POTCAR

```powershell
uv run --frozen magvasp potcar scan `
  --root "C:\path\to\potpaw_PBE.54" `
  --family PBE_54 `
  --output var\potcar_registry.json
```

不要把 POTCAR、registry、计算结果或第三方 VASPKIT 发行包提交到 Git。

### 模型

```powershell
Copy-Item config\agent\llm.example.json config\agent\llm.json
$env:MAGVASP_LLM_API_KEY = "your-key"
```

密钥只从环境变量读取；`config/agent/llm.json` 只写模型服务配置。

### 集群

1. 管理员在受控环境中填写 `config/execution/site.team.json`，替换所有 `example` 值。
2. 用户复制 `config/execution/identity.example.json` 为 `identity.json`，填写自己的账号和连接方式。
3. 保持 `execution_enabled=false`，先完成登录、probe 和资源规则确认。
4. 合成本机 profile：

```powershell
uv run --frozen magvasp execution compose `
  --site config\execution\site.team.json `
  --identity config\execution\identity.json `
  --output config\execution\cluster.json
```

`identity.json`、`cluster.json`、`.env` 和 `var/` 都是本机文件。远端目录必须包含当前用户自己的账号路径。

## 启动网页

```powershell
uv run --extra agent --extra web --frozen magvasp web `
  --profile config\execution\cluster.json
```

浏览器访问 `http://127.0.0.1:8765`。网页不绑定局域网或公网。

## 远程执行前

```powershell
uv run --frozen magvasp execution login `
  --profile config\execution\cluster.json `
  --connect --interactive
uv run --frozen magvasp execution probe `
  --profile config\execution\cluster.json --node <approved-node>
```

真实启动还需要输入审核、资源批准、MPI smoke、受控传输和本次 launch 批准。程序不接受任意 shell，不提供自动抢占、自动重试或取消他人作业。

## 安全检查

```powershell
git diff --check
git status --short
git grep -n -I -E 'MAGVASP_LLM_API_KEY=|BEGIN (OPENSSH|RSA)|password='
```

任何凭据一旦进入 Git 历史，应立即撤销并按组织流程清理历史。
