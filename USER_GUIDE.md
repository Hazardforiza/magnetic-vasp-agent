# Magnetic VASP Agent 用户使用指南

## 目录

1. [快速开始](#快速开始)
2. [系统要求与安装](#系统要求与安装)
3. [首次配置](#首次配置)
4. [Web 界面使用](#web-界面使用)
5. [VASPKIT 后处理](#vaspkit-后处理)
6. [磁性分析工具](#磁性分析工具)
7. [课题组协作功能](#课题组协作功能)
8. [集群配置](#集群配置)
9. [常见问题](#常见问题)

---

## 快速开始

### 系统要求

- **操作系统**: Linux / macOS / Windows (WSL2)
- **Python 版本**: 3.11+
- **包管理器**: [uv](https://github.com/astral-sh/uv) (推荐) 或 pip
- **VASP 赝势库**: 本地 POTCAR 文件（用于输入文件生成）

### 最简安装流程

```bash
# 1. 克隆仓库
git clone https://github.com/Hazardforiza/magnetic-vasp-agent.git
cd magnetic-vasp-agent

# 2. 安装依赖（推荐使用 uv）
uv sync --all-extras

# 3. 配置 POTCAR 路径
magvasp scan-potcar /path/to/your/potpaw_PBE

# 4. 配置 LLM API（可选，用于 Agent 功能）
export OPENAI_API_KEY="your-api-key"

# 5. 启动 Web 界面
magvasp web
```

打开浏览器访问 `http://localhost:8000` 即可使用。

---

## 系统要求与安装

### 完整功能安装

如需使用全部功能（Web 界面、VASPKIT 集成、磁性分析、课题组协作）：

```bash
uv sync --all-extras
```

### 按需模块化安装

如果只需要特定功能，可以按需安装：

```bash
# 仅 Agent 核心功能
uv sync --extra agent

# Agent + Web 界面
uv sync --extra agent --extra web

# Agent + VASPKIT 后处理
uv sync --extra agent --extra vaspkit

# Agent + 磁性分析
uv sync --extra agent --extra magnetic

# 完整开发环境（含测试工具）
uv sync --all-extras --extra dev
```

### 依赖说明

- **核心依赖**: pymatgen、ase、LangChain、FastAPI
- **Web 界面**: uvicorn、jinja2、aiofiles
- **VASPKIT**: matplotlib、pandas、scipy
- **磁性分析**: numpy、pymatgen 磁性模块
- **课题组协作**: YAML、JSON 配置管理

---

## 首次配置

### 1. POTCAR 赝势库配置

#### 扫描本地赝势库

```bash
# 扫描 VASP 赝势库目录
magvasp scan-potcar /path/to/potpaw_PBE

# 查看已配置的赝势库
magvasp list-potcar
```

#### 配置文件位置

扫描后的配置保存在 `~/.magvasp/potcar_config.json`。手动编辑示例：

```json
{
  "potcar_paths": [
    "/home/user/vasp/potpaw_PBE",
    "/mnt/shared/potpaw_GGA"
  ],
  "default_potcar_set": "PBE"
}
```

### 2. LLM API 配置

#### OpenAI API

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选，国内镜像可修改
```

#### 其他兼容 OpenAI 格式的 API

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://your-llm-service.com/v1"
export OPENAI_MODEL="gpt-4o-mini"  # 可选，默认模型
```

#### 配置文件方式

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

### 3. 集群身份配置

#### 个人身份文件 (`identity.json`)

在项目根目录创建 `identity.json`：

```json
{
  "user": "your-username",
  "email": "your-email@example.com",
  "ssh_key_path": "~/.ssh/id_rsa",
  "default_site": "local"
}
```

#### 课题组站点配置 (`site.team.json`)

在项目根目录创建 `site.team.json`（管理员维护）：

```json
{
  "sites": [
    {
      "name": "local",
      "host": "localhost",
      "type": "local"
    },
    {
      "name": "hpc-cluster",
      "host": "login.hpc.edu.cn",
      "port": 22,
      "type": "slurm",
      "work_dir": "/home/{user}/vasp-work",
      "modules": ["intel/2023", "vasp/6.4.0"]
    }
  ]
}
```

#### 合成最终配置

```bash
magvasp compose
```

生成 `site.json`（包含个人身份 + 课题组站点）。

---

## Web 界面使用

### 启动 Web 服务

```bash
# 方式 1：命令行启动
magvasp web

# 方式 2：指定端口
magvasp web --port 8080

# 方式 3：Windows 批处理脚本
start-web.bat
```

启动后访问 `http://localhost:8000`。

### 主要功能模块

#### 1. 知识库浏览

- **路径**: `/knowledge`
- **功能**: 浏览课题组经验库、参考案例、技术文档
- **操作**: 点击条目查看详情、下载附件、搜索关键词

#### 2. 输入文件生成与审核

- **路径**: `/generate`
- **功能**: 基于结构文件（CIF/POSCAR）生成 VASP 输入文件
- **流程**:
  1. 上传结构文件（支持 mCIF 磁性结构）
  2. 选择计算类型（能带、态密度、磁性优化等）
  3. Agent 自动生成 INCAR、KPOINTS
  4. 人工审核确认（强制审核，避免计算错误）
  5. 下载输入文件包

#### 3. Agent 会话运行

- **路径**: `/agent`
- **功能**: 与 Agent 对话，执行复杂工作流
- **示例**:
  - "帮我计算 Fe₃O₄ 的磁各向异性能"
  - "分析 OUTCAR 中的磁矩变化"
  - "比较两个磁构型的能量差"

#### 4. 计算结果导入

- **路径**: `/results`
- **功能**: 上传 VASP 输出文件（OUTCAR、vasprun.xml），自动提取结果并存入数据库
- **支持格式**: OUTCAR、vasprun.xml、OSZICAR

---

## VASPKIT 后处理

### DOS（态密度）绘图

#### 绘制总态密度

```bash
magvasp postprocess plot-dos --tdos TDOS.dat
```

#### 绘制分波态密度（PDOS）

```bash
magvasp postprocess plot-dos --pdos PDOS.dat --tdos TDOS.dat
```

#### 自定义绘图参数

```bash
magvasp postprocess plot-dos \
  --tdos TDOS.dat \
  --output dos_plot.png \
  --energy-range -5 5 \
  --dpi 300
```

### 能带（Band Structure）绘图

#### 基础能带绘图

```bash
magvasp postprocess plot-band \
  --band BAND.dat \
  --labels KLABELS
```

#### 带投影能带（Fat Band）

```bash
magvasp postprocess plot-band \
  --band BAND.dat \
  --labels KLABELS \
  --projected PBAND.dat
```

### VASPKIT 任务执行

#### 查看可用任务

```bash
magvasp vaspkit list-tasks
```

#### 执行特定任务

```bash
# 任务 303：态密度后处理
magvasp vaspkit run-task 303 --input-dir ./calc-dos

# 任务 211：能带后处理
magvasp vaspkit run-task 211 --input-dir ./calc-band

# 任务 111：提取晶格参数
magvasp vaspkit run-task 111 --input-dir ./calc-relax
```

---

## 磁性分析工具

### 磁构型能量比较

比较不同磁序（FM/AFM/FiM）的能量：

```bash
magnetic-analysis compare \
  --structures FM.vasp AFM1.vasp AFM2.vasp \
  --labels FM AFM1 AFM2 \
  --output comparison.json
```

输出示例：

```json
{
  "ground_state": "AFM1",
  "energies": {
    "FM": 0.0,
    "AFM1": -0.034,
    "AFM2": -0.018
  }
}
```

### 磁各向异性能（MAE）计算

```bash
magnetic-analysis mae \
  --structure Fe.vasp \
  --soc-dirs "001 100" \
  --output mae_result.json
```

计算不同磁化方向的能量差：

```json
{
  "mae_meV_per_atom": 0.52,
  "easy_axis": "001",
  "hard_axis": "100"
}
```

### 交换相互作用提取

基于多磁构型能量拟合 Heisenberg 模型 J 参数：

```bash
magnetic-analysis exchange \
  --config magnetic_configs.json \
  --output exchange_J.json
```

`magnetic_configs.json` 格式：

```json
{
  "configurations": [
    {"label": "FM", "file": "FM.vasp", "energy": -10.5},
    {"label": "AFM", "file": "AFM.vasp", "energy": -10.8}
  ]
}
```

---

## 课题组协作功能

### 经验库管理

#### 添加经验条目

```bash
magvasp lesson add \
  --title "Fe3O4磁性优化经验" \
  --content "ISPIN=2, MAGMOM建议..." \
  --tags "磁性,Fe3O4,优化"
```

#### 搜索经验库

```bash
magvasp lesson search --keyword "磁性"
magvasp lesson search --tag "Fe3O4"
```

#### 导出经验库

```bash
magvasp lesson export --output knowledge_base.json
```

### 项目资源管理

#### 创建项目

```bash
magvasp project create \
  --name "Fe3O4-MAE研究" \
  --description "Fe3O4磁各向异性能计算" \
  --members "user1,user2"
```

#### 添加项目资源

```bash
magvasp project add-resource \
  --project "Fe3O4-MAE研究" \
  --file ./INCAR.template \
  --type template \
  --description "MAE计算INCAR模板"
```

#### 查看项目资源

```bash
magvasp project list-resources --project "Fe3O4-MAE研究"
```

### 批量作业管理

#### 提交批量作业

```bash
magvasp batch submit \
  --config batch_config.yaml \
  --site hpc-cluster
```

`batch_config.yaml` 格式：

```yaml
jobs:
  - name: Fe3O4-FM
    structure: Fe3O4-FM.vasp
    calc_type: relax
  - name: Fe3O4-AFM
    structure: Fe3O4-AFM.vasp
    calc_type: relax
```

#### 查看作业状态

```bash
magvasp batch status --job-id batch-20260923-001
```

---

## 集群配置

### 站点模板结构

在 `site.team.json` 中定义集群站点：

```json
{
  "sites": [
    {
      "name": "tianhe2",
      "host": "login.nscc-gz.cn",
      "port": 22,
      "type": "pbs",
      "work_dir": "/WORK/{user}/vasp",
      "modules": ["intel/2019", "vasp/5.4.4"],
      "queue_config": {
        "default_queue": "normal",
        "max_walltime": "24:00:00",
        "max_nodes": 16
      }
    }
  ]
}
```

### 个人身份配置

在 `identity.json` 中配置个人信息：

```json
{
  "user": "zhangsan",
  "email": "zhangsan@example.com",
  "ssh_key_path": "~/.ssh/id_rsa_hpc",
  "default_site": "tianhe2",
  "custom_modules": ["gcc/9.3.0"]
}
```

### SSH 配置最佳实践

#### 1. 配置 SSH 密钥登录

```bash
# 生成密钥对
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa_hpc

# 上传公钥到集群
ssh-copy-id -i ~/.ssh/id_rsa_hpc.pub user@login.hpc.edu.cn
```

#### 2. 配置 SSH 免密登录（`~/.ssh/config`）

```
Host hpc-cluster
    HostName login.hpc.edu.cn
    User zhangsan
    Port 22
    IdentityFile ~/.ssh/id_rsa_hpc
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

#### 3. 测试连接

```bash
ssh hpc-cluster "echo 'Connection OK'"
```

---

## 常见问题

### 1. 安装相关

#### Q: `uv sync` 失败，提示缺少依赖

**A**: 确保 Python 版本 ≥ 3.11，并更新 uv：

```bash
python --version  # 应显示 3.11.x 或更高
pip install --upgrade uv
uv sync --all-extras
```

#### Q: Windows 上安装失败

**A**: 推荐使用 WSL2 或安装 Visual Studio Build Tools：

```powershell
# 或在 PowerShell 中
pip install --upgrade pip setuptools wheel
uv sync --all-extras
```

### 2. POTCAR 配置

#### Q: 扫描 POTCAR 后提示找不到文件

**A**: 检查路径是否正确：

```bash
# 正确路径应包含元素目录，如：
/path/to/potpaw_PBE/Fe/POTCAR
/path/to/potpaw_PBE/O/POTCAR

# 扫描时使用上级目录
magvasp scan-potcar /path/to/potpaw_PBE
```

#### Q: 生成输入文件时提示 POTCAR 缺失

**A**: 确认赝势库中有该元素：

```bash
magvasp list-potcar | grep "Fe"
```

如果缺失，需补充对应元素的 POTCAR 文件。

### 3. LLM API 配置

#### Q: Agent 调用失败，提示 API Key 错误

**A**: 检查环境变量是否生效：

```bash
echo $OPENAI_API_KEY

# 如果为空，重新设置
export OPENAI_API_KEY="sk-..."
```

#### Q: 使用国内镜像 API 时超时

**A**: 修改 `OPENAI_BASE_URL`：

```bash
export OPENAI_BASE_URL="https://your-mirror.com/v1"
```

### 4. Web 界面

#### Q: 启动 Web 服务后无法访问

**A**: 检查端口占用：

```bash
# Linux/macOS
lsof -i :8000

# Windows
netstat -ano | findstr :8000

# 更换端口
magvasp web --port 8080
```

#### Q: 上传文件后 Agent 无响应

**A**: 检查后台日志：

```bash
tail -f ~/.magvasp/logs/web.log
```

### 5. 集群执行

#### Q: SSH 连接集群失败

**A**: 检查 SSH 配置和密钥权限：

```bash
# 测试连接
ssh -vvv user@login.hpc.edu.cn

# 修复密钥权限
chmod 600 ~/.ssh/id_rsa_hpc
chmod 644 ~/.ssh/id_rsa_hpc.pub
```

#### Q: 作业提交后一直处于队列

**A**: 检查集群队列状态：

```bash
# SLURM
squeue -u $USER

# PBS
qstat -u $USER
```

### 6. 磁性分析

#### Q: MAE 计算结果为负值

**A**: 负值表示能量降低是正常的，取绝对值即为 MAE：

```python
mae = abs(E_hard - E_easy)
```

#### Q: 交换相互作用拟合失败

**A**: 检查磁构型数量是否足够（至少 3 个）且能量差异明显。

### 7. VASPKIT 后处理

#### Q: 绘图时提示缺少 KLABELS 文件

**A**: 确保 VASPKIT 任务 211 正确执行：

```bash
# 重新生成
magvasp vaspkit run-task 211 --input-dir ./calc-band

# 检查输出
ls -lh KLABELS BAND.dat
```

### 8. 测试相关

#### Q: 运行测试失败

**A**: 确保开发依赖已安装：

```bash
uv sync --all-extras --extra dev
uv run pytest tests/ -v
```

#### Q: 测试报告显示 warnings

**A**: warnings 通常不影响功能，可忽略。如需消除：

```bash
uv run pytest tests/ -v --disable-warnings
```

---

## 技术支持

- **GitHub Issues**: [https://github.com/Hazardforiza/magnetic-vasp-agent/issues](https://github.com/Hazardforiza/magnetic-vasp-agent/issues)
- **文档**: 参见 `README.md` 和 `docs/` 目录
- **示例**: 参见 `examples/` 和 `Reference_Demo/` 目录

---

## 许可证

本项目采用 MIT 许可证。详见 `LICENSE` 文件。
