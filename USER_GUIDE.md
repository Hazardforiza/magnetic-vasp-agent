# Magnetic VASP Agent - 用户配置与使用指南

## 📋 前置要求

- **Python**: 3.11 或更高版本
- **包管理器**: [uv](https://github.com/astral-sh/uv) (推荐) 或 pip
- **SSH 访问**: 可访问的计算集群，配置好 SSH 密钥认证
- **POTCAR 库**: 有效的 VASP 赝势文件库（需要 VASP 许可证）
- **LLM API**: OpenAI API 密钥或兼容的 API 端点

## 🚀 快速开始

### 1. 克隆仓库并安装依赖

```bash
git clone https://github.com/Hazardforiza/magnetic-vasp-agent.git
cd magnetic-vasp-agent
uv sync --frozen
```

### 2. 配置 POTCAR 赝势库

扫描你的 POTCAR 文件目录并生成索引：

```bash
uv run magvasp potcar scan \
  --root "C:/POTCAR/PBE.54" \
  --family PBE_54 \
  --output var/potcar_registry.json
```

支持的赝势族：
- `PBE_54`: 标准 PBE 泛函（54 元素）
- `PBE_52`: 旧版 PBE（52 元素）
- `LDA`: LDA 泛函

### 3. 配置 LLM API

**方式一：环境变量（推荐）**

在 shell 中设置：

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_API_BASE="https://api.openai.com/v1"  # 可选
```

或创建 `.env` 文件（已在 `.gitignore` 中，不会提交）：

```
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1
```

**方式二：配置文件**

编辑 `config/agent/llm.json`（不要提交包含密钥的版本）：

```json
{
  "provider": "openai",
  "model": "gpt-4",
  "api_key": "${OPENAI_API_KEY}",
  "base_url": "https://api.openai.com/v1",
  "temperature": 0.1
}
```

### 4. 配置集群连接

#### 4.1 设置 SSH 密钥

确保你的 SSH 密钥已配置并可以免密登录集群：

```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa
ssh-copy-id your_username@login.hpc.edu
ssh your_username@login.hpc.edu  # 测试连接
```

#### 4.2 配置集群信息

创建或编辑 `config/execution/cluster.json`：

```json
{
  "default": {
    "host": "login.hpc.edu",
    "port": 22,
    "username": "your_username",
    "scheduler": "slurm",
    "vasp_cmd": "mpirun -np {cores} vasp_std",
    "cores_per_node": 24,
    "max_nodes": 4,
    "partition": "regular",
    "vasp_root": "/opt/vasp/6.4.0"
  }
}
```

**字段说明**：
- `host`: 集群登录节点地址
- `username`: SSH 用户名
- `scheduler`: 作业调度器类型（`slurm` 或 `pbs`）
- `vasp_cmd`: VASP 执行命令模板，`{cores}` 会被替换为实际核心数
- `cores_per_node`: 每节点核心数
- `partition`: 作业队列/分区名称

#### 4.3 配置用户身份信息

创建或编辑 `config/execution/identity.json`：

```json
{
  "name": "张三",
  "affiliation": "计算材料科学组",
  "email": "zhangsan@university.edu",
  "ssh_key_path": "~/.ssh/id_rsa",
  "default_cluster": "default"
}
```

**⚠️ 安全提示**：此文件已在 `.gitignore` 中，不会被提交到 Git。

## 💻 基本使用

### 查看可用工具

```bash
uv run magvasp agent tools
```

### 启动交互式 Agent

```bash
uv run magvasp agent chat \
  --goal "计算 FeO 的结构弛豫" \
  --config config/agent/llm.json
```

### 命令行执行（单次任务）

```bash
uv run magvasp agent run \
  --task "生成 Fe2O3 的 INCAR 文件用于静态计算" \
  --output results/
```

### 批量计算工作流

```bash
uv run magvasp workflow batch \
  --input structures/*.cif \
  --task-type relax \
  --output batch_results/
```

## 🔄 人机协作审核流程

Agent 在执行关键操作前会请求人工审批：

1. **INCAR 生成审核**：生成 VASP 输入文件后，会展示差异让你确认
2. **作业提交审核**：提交集群作业前，会显示作业脚本让你批准
3. **mCIF 结构审核**：从磁性 CIF 生成 POSCAR 时强制人工检查

审核示例输出：

```
[APPROVAL REQUIRED] Submit VASP job to cluster?

Job script:
  #!/bin/bash
  #SBATCH --nodes=1
  #SBATCH --ntasks=24
  mpirun -np 24 vasp_std

Approve? [y/n]:
```

## 📂 目录结构说明

```
magnetic-vasp-agent/
├── config/              # 配置文件（部分不提交到 Git）
│   ├── agent/          # Agent 和 LLM 配置
│   └── execution/      # 集群和身份配置（敏感信息）
├── var/                # 运行时数据（不提交）
│   ├── potcar_registry.json  # POTCAR 索引
│   └── workflows/      # 工作流执行记录
├── examples/           # 示例输入文件
│   ├── structures/     # CIF 示例
│   └── workflows/      # 工作流模板
└── tests/              # 单元测试
```

## 🔧 高级配置

### 自定义 INCAR 模板

在 `config/templates/` 下创建自定义模板：

```python
# config/templates/custom_relax.py
def get_incar_template(system_info):
    return {
        "PREC": "Accurate",
        "ENCUT": 600,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        # ... 更多参数
    }
```

### Skills 扩展

Agent 自动加载 `skills/` 目录下的技能模块：

```python
# skills/my_custom_skill.py
from magvasp.agent import Tool

class MyCustomTool(Tool):
    name = "my_tool"
    description = "做某件事"
    
    def execute(self, **kwargs):
        # 实现逻辑
        pass
```

### Reference_Demo 只读参考

`Reference_Demo/` 目录包含已验证的计算案例，Agent 可以参考但不能修改：

- 结构文件
- 经过验证的 INCAR 参数组合
- 收敛性测试记录

## ⚠️ 故障排除

### SSH 连接失败

```
Error: Failed to connect to cluster
```

**解决方案**：
1. 检查 SSH 密钥权限：`chmod 600 ~/.ssh/id_rsa`
2. 测试手动连接：`ssh -vvv your_username@login.hpc.edu`
3. 确认 `config/execution/cluster.json` 中的 `host` 和 `username` 正确

### POTCAR 文件未找到

```
Error: POTCAR not found for element Fe
```

**解决方案**：
1. 重新扫描 POTCAR 库：`uv run magvasp potcar scan ...`
2. 检查 `var/potcar_registry.json` 是否存在且包含对应元素
3. 确认 POTCAR 文件命名符合 VASP 标准（如 `POTCAR_PBE_Fe`）

### 集群作业提交失败

```
Error: sbatch command not found
```

**解决方案**：
1. 确认集群使用的调度器类型（Slurm 或 PBS）
2. 更新 `config/execution/cluster.json` 中的 `scheduler` 字段
3. 检查登录节点是否可以访问调度器命令

### LLM API 调用超时

```
Error: Timeout calling OpenAI API
```

**解决方案**：
1. 检查网络连接和防火墙设置
2. 如果使用代理，设置 `HTTP_PROXY` 和 `HTTPS_PROXY` 环境变量
3. 尝试更换 API 端点（`base_url` 配置项）

### VASP 计算结果不收敛

**解决方案**：
1. 检查 `OUTCAR` 中的警告信息
2. 调整 INCAR 参数（如增加 `NELM`、调整 `EDIFF`）
3. 参考 `Reference_Demo/` 中的类似体系案例
4. 使用 `magvasp analyze convergence` 工具分析收敛性

## 🔒 安全与隐私

- **不要提交敏感信息**：`.gitignore` 已配置忽略 API 密钥、SSH 配置、POTCAR 文件
- **本地运行**：Agent 是本地桌面工具，所有数据留在本地或你的集群
- **人工审核**：关键操作需要人工批准，LLM 无法绕过
- **只读 Skills**：Agent 只能读取 `skills/` 和 `Reference_Demo/`，不能修改

## 📚 更多资源

- **开发者文档**：[README.md](README.md)
- **部署指南**：[DEPLOYMENT.md](docs/DEPLOYMENT.md)
- **API 参考**：[docs/API.md](docs/API.md)
- **问题反馈**：[GitHub Issues](https://github.com/Hazardforiza/magnetic-vasp-agent/issues)

## 📝 许可证

本项目采用 MIT 许可证。注意：VASP 和 POTCAR 文件受独立的商业许可证约束。
