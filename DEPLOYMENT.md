# Magnetic VASP Agent 部署指南

本指南适用于在新机器上部署本项目的**稳定版本**（最近干净提交）。

## 部署前提条件

### 系统要求

- **操作系统**：Windows 10/11 或 Linux
- **Python 版本**：3.11 或更高
- **网络要求**：能访问集群跳板机（`mu01`）
  - 工位直连：`192.168.164.203:7722`
  - 或通过本机隧道：`127.0.0.1:17722`

### 必需的外部资源（每台机器独立配置）

1. **授权的 VASP POTCAR 库**（版权保护，无法共享）
2. **大模型 API 密钥**（OpenAI 或兼容接口）
3. **集群 SSH 账号**（mu01 跳板机的用户名和密码）

---

## 第一步：安装 Python 环境

### 安装 Python 3.11

**Windows**：
```powershell
# 从 https://www.python.org/downloads/ 下载并安装 Python 3.11
# 安装时勾选 "Add Python to PATH"
python --version  # 确认版本 >= 3.11
```

**Linux**：
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv

# 或使用 pyenv
pyenv install 3.11
pyenv global 3.11
```

### 安装 uv 包管理器

```powershell
# Windows PowerShell
irm https://astral.sh/uv/install.ps1 | iex

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

验证安装：
```powershell
uv --version
```

---

## 第二步：克隆仓库

```powershell
# 从 GitHub 克隆（替换为实际仓库地址）
git clone https://github.com/<your-org>/magnetic-vasp-agent.git
cd magnetic-vasp-agent

# 检查当前版本
git log --oneline -1
```

---

## 第三步：安装项目依赖

```powershell
# 安装所有依赖（agent + web）
uv sync --extra agent --extra web --frozen

# 验证安装
uv run --frozen magvasp --help
```

---

## 第四步：配置 POTCAR 赝势库

### 准备 POTCAR 目录

确保你有授权的 VASP POTCAR 库，通常结构如下：
```
/path/to/vasp_potcars/
├── PBE.54/
│   ├── Al/
│   │   └── POTCAR
│   ├── Cu/
│   │   └── POTCAR
│   └── ...
```

### 扫描并注册 POTCAR

```powershell
# 扫描 PBE.54 势函数集（根据实际路径调整）
uv run --frozen magvasp potcar scan `
  --root "D:\vasp_potcars\PBE.54" `
  --collection PBE.54 `
  --xc PBE

# 验证注册表
uv run --frozen magvasp potcar list
```

**重要**：
- POTCAR 文件受版权保护，不能通过网络共享
- 每台机器必须独立扫描各自授权的库
- 扫描结果保存在本地 `var/potcar_registry.json`

---

## 第五步：配置大模型 API

### 创建 LLM 配置文件

```powershell
# 创建配置目录
New-Item -ItemType Directory -Force -Path config\agent

# 创建配置文件 config/agent/llm.json
@"
{
  "provider": "openai",
  "model": "gpt-4",
  "api_base": "https://api.openai.com/v1",
  "temperature": 0.0,
  "max_tokens": 4096
}
"@ | Out-File -Encoding utf8 config\agent\llm.json
```

### 配置 API 密钥（环境变量）

**方式一：临时设置（当前 PowerShell 会话）**
```powershell
$env:MAGVASP_LLM_API_KEY = "sk-your-api-key-here"
```

**方式二：持久化（推荐）**

创建 `.env` 文件（**不要提交到 Git**）：
```powershell
@"
MAGVASP_LLM_API_KEY=sk-your-api-key-here
"@ | Out-File -Encoding utf8 .env

# 加载到环境变量
Get-Content .env | ForEach-Object {
  if ($_ -match '^(MAGVASP_LLM_API_KEY)=(.*)$') {
    Set-Item -Path "env:$($matches[1])" -Value $matches[2]
  }
}
```

**重要**：
- **绝对不要**把 API 密钥写进代码或配置文件并提交到 Git
- `.env` 文件已在 `.gitignore` 中，不会被提交

---

## 第六步：配置集群身份

### 复制身份模板

```powershell
Copy-Item config\execution\identity.example.json config\execution\identity.json
```

### 编辑身份配置

打开 `config/execution/identity.json`，修改以下字段：

```json
{
  "user": "your_cluster_username",
  "connection": "office",
  "execution_enabled": false,
  "remote_root": "/home/your_cluster_username/magvasp/runs"
}
```

**字段说明**：
- `user`：你在 mu01 集群上的用户名
- `connection`：选择连接方式
  - `"office"`：工位直连 `192.168.164.203:7722`
  - `"tunnel"`：通过本机隧道 `127.0.0.1:17722`
- `execution_enabled`：初始设为 `false`，等组内放行后再改为 `true`
- `remote_root`：**必须**是你自己的家目录路径，不要用别人的

### 合成运行配置

```powershell
uv run --frozen magvasp execution compose `
  --site config\execution\site.team.json `
  --identity config\execution\identity.json `
  --output config\execution\cluster.json `
  --overwrite
```

**重要**：
- `cluster.json` 是本机生成的，**不要**复制别人的
- SSH 密码只在实际连接时输入，不会保存在文件中

---

## 第七步：验证安装

### 运行诊断命令

```powershell
uv run --frozen magvasp doctor
```

**检查项**：
- ✅ Python 版本
- ✅ uv 环境
- ✅ POTCAR 注册表
- ✅ LLM 配置
- ✅ 集群配置

### 测试基础功能

**1. 查看可用工具**
```powershell
uv run --frozen magvasp agent tools
```

**2. 解析示例结构**
```powershell
uv run --frozen magvasp structure parse `
  tests\fixtures\structures\Mn3Al\POSCAR
```

**3. 测试 POTCAR 选择**
```powershell
uv run --frozen magvasp potcar select `
  --elements Mn Al `
  --task scf `
  --magnetic collinear `
  --xc PBE
```

---

## 第八步：启动网页界面（可选）

### 启动服务

```powershell
uv run --extra web --frozen magvasp web
```

浏览器会自动打开 `http://127.0.0.1:8765`

### 或使用快捷方式（Windows）

双击仓库根目录的 `start-web.bat`

**网页功能**：
- 上传和解析 mCIF 文件
- 生成和审核 VASP 输入包
- 启动 Agent 会话（需要完成第五步的 LLM 配置）
- 导入和查看计算结果

**注意**：
- 网页仅监听 `127.0.0.1`，只能本机访问
- 不是共享的在线服务
- 每个组员需在各自电脑上启动

---

## 第九步：测试 Agent 功能（可选）

### 准备测试

确保已完成第五步（LLM 配置）和第六步（集群配置）。

### 运行简单测试

```powershell
# 加载环境变量（如果还没加载）
Get-Content .env | ForEach-Object {
  if ($_ -match '^(MAGVASP_LLM_API_KEY)=(.*)$') {
    Set-Item -Path "env:$($matches[1])" -Value $matches[2]
  }
}

# 测试 Agent（不涉及真实集群执行）
uv run --frozen magvasp agent chat `
  --goal "列出可用的磁性计算工具" `
  --config config\agent\llm.json `
  --output var\test-chat.json
```

### 真实集群执行（谨慎）

**在新机器上不要立即尝试真实集群执行**，建议：

1. 先在本机验证 POTCAR、结构解析、输入生成
2. 与组内确认集群资源分配
3. 将 `identity.json` 中的 `execution_enabled` 改为 `true`
4. 小规模测试（如 `execution smoke`）
5. 逐步验证 transfer 和 launch

---

## 常见问题

### Q1: POTCAR 扫描失败

**问题**：`magvasp potcar scan` 报错

**解决**：
- 确认 POTCAR 目录结构正确
- 确认有读取权限
- 检查文件编码（应为 ASCII 或 UTF-8）

### Q2: LLM API 调用失败

**问题**：Agent 无法连接大模型

**解决**：
- 确认 `MAGVASP_LLM_API_KEY` 环境变量已设置
- 检查 `config/agent/llm.json` 中的 `api_base` 是否正确
- 测试网络连接：`curl https://api.openai.com/v1/models`

### Q3: SSH 连接集群失败

**问题**：`execution compose` 或 `execution probe` 报错

**解决**：
- 确认网络能访问跳板机（`ping 192.168.164.203` 或本地隧道）
- 确认用户名和密码正确
- 检查 `identity.json` 中的 `connection` 字段（`office` 或 `tunnel`）

### Q4: 网页无法启动

**问题**：`magvasp web` 报端口占用

**解决**：
```powershell
# 检查端口占用
netstat -ano | findstr :8765

# 或指定其他端口
uv run --frozen magvasp web --port 8766
```

### Q5: 测试失败

**问题**：`pytest` 或某些功能不工作

**解决**：
- 确认依赖完整安装：`uv sync --extra agent --extra web --frozen`
- 查看日志：`var/logs/`
- 运行诊断：`uv run --frozen magvasp doctor`

---

## 安全注意事项

### 不要共享的内容（仅本机）

- `.env` 文件（API 密钥）
- `config/execution/identity.json`（集群账号）
- `config/execution/cluster.json`（合成的运行配置）
- `var/` 目录（计算结果、日志）
- POTCAR 文件或注册表

### 可以共享的内容（通过 Git）

- 源代码
- 配置模板（`*.example.json`）
- 文档（`docs/`）
- 全组站点配置（`config/execution/site.team.json`）

### Git 提交前检查

```powershell
# 确认没有敏感信息
git status
git diff

# .gitignore 应该已经排除敏感文件
```

---

## 下一步建议

### 部署成功后

1. **验证基础功能**：
   - POTCAR 扫描和注册
   - 结构解析（POSCAR/mCIF）
   - 输入包生成和校验

2. **不要立即执行**：
   - 等待组内确认资源分配
   - 先用 `execution smoke` 小规模测试
   - 逐步验证 transfer 和 launch

3. **参考现有案例**：
   - 阅读 `docs/knowledge/member_setup.md`
   - 查看 README 中的"未来目标"章节
   - 了解已完成的里程碑（Steps 1-12）

### 遇到问题

- 查看日志：`var/logs/`
- 运行诊断：`magvasp doctor`
- 参考文档：`docs/`
- 联系组内成员

---

## 附录：命令速查表

```powershell
# 安装
uv sync --extra agent --extra web --frozen

# 诊断
uv run --frozen magvasp doctor

# POTCAR
uv run --frozen magvasp potcar scan --root <path> --collection PBE.54 --xc PBE
uv run --frozen magvasp potcar list

# 结构
uv run --frozen magvasp structure parse <POSCAR>

# 集群配置
uv run --frozen magvasp execution compose --site <site.json> --identity <identity.json> --output cluster.json

# Agent
uv run --frozen magvasp agent chat --goal "<目标>" --config config\agent\llm.json

# 网页
uv run --frozen magvasp web

# 测试
uv run --frozen pytest tests/
```

---

**版本信息**：本指南适用于最近的干净提交版本（不包含工作区未提交改动）

**最后更新**：2026-10-20
