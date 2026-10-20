# Magnetic VASP Agent

面向磁性材料 VASP 第一性原理计算的智能 Agent 系统。

## 🎯 项目简介

Magnetic VASP Agent 是一个基于 LangGraph 的 AI Agent，用于辅助磁性材料的 VASP（Vienna Ab initio Simulation Package）第一性原理计算。该项目采用**确定性计算内核 + RAG 知识库 + LLM Agent 编排**的三层架构，在保证计算准确性的同时提供智能化的工作流自动化和人工审核机制。

### 核心特性

- **确定性计算内核**：POTCAR/POSCAR/INCAR/KPOINTS 生成，保证输入文件的正确性
- **RAG 知识治理**：整合 VASP Wiki、课题组文档、参考案例的检索增强生成
- **人工审核工作流**：所有关键操作（集群执行、文件修改）需人工批准
- **真实集群集成**：SSH + MPI 跨节点作业提交与监控
- **模板化工作流**：支持课题组共享的 mCIF 模板和计算工作流

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────┐
│         LLM Agent 编排层 (LangGraph)             │
│  - 工具调用协议 (approval state machine)         │
│  - 多 Agent 编排 (structure → workflow → exec)  │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│           RAG 知识库 (Qdrant + BM25)             │
│  - VASP Wiki 爬取与索引                          │
│  - 课题组 Skills / Reference_Demo 只读集成       │
│  - 证据包生成 (Evidence Package)                 │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│        确定性计算内核 (Python Core)               │
│  - POTCAR 扫描与元素查找                         │
│  - POSCAR 结构解析 (磁矩、元素、晶格)             │
│  - INCAR 模板化生成 (磁性 ENCUT/KPOINTS 推荐)   │
│  - Workflow 规划器 (relax → static → dos)        │
│  - 集群执行运行时 (SSH/MPI, Slurm/PBS)           │
│  - VASP 输出解析 (OSZICAR/OUTCAR/vasprun.xml)   │
└─────────────────────────────────────────────────┘
```

## 📦 技术栈

- **Python**: 3.11+
- **依赖管理**: `uv` (快速、可重现的依赖锁定)
- **核心框架**: LangGraph, LangChain, Pydantic V2
- **向量检索**: Qdrant (本地模式), Rank-BM25
- **SSH/集群**: Fabric, Paramiko
- **测试框架**: pytest

## 🚀 快速开始

### 1. 环境要求

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) 包管理器
- SSH 密钥（用于连接计算集群）
- POTCAR 库（VASP 官方赝势文件）

### 2. 克隆仓库

```bash
git clone https://github.com/Hazardforiza/magnetic-vasp-agent.git
cd magnetic-vasp-agent
```

### 3. 安装依赖

```bash
# 开发环境（包含测试工具）
uv sync --extra dev --frozen

# 生产环境
uv sync --frozen
```

### 4. 运行测试

```bash
uv run --frozen pytest
```

### 5. 快速验证

```bash
# 查看所有可用的 Agent 工具
uv run --frozen magvasp agent tools

# 扫描本地 POTCAR 库（首次使用必须）
uv run --frozen magvasp potcar scan --root "C:/POTCAR/PBE.54" --family PBE_54 --output var/potcar_registry.json
```

## 📚 文档导航

- **[USER_GUIDE.md](USER_GUIDE.md)** - 用户配置与使用指南（首次部署必读）
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - 跨机器部署指南
- **[AGENTS.md](AGENTS.md)** - Agent 开发规范
- **[docs/](docs/)** - 详细技术文档

## 🛠️ 开发里程碑

### 已完成（步骤 1-16）

- **步骤 1-4**: POTCAR 扫描、POSCAR 解析、INCAR 模板、KPOINTS 生成
- **步骤 5-7**: Workflow 规划器、执行计划生成、批处理器
- **步骤 8-9**: SSH 集群抽象、Slurm/PBS 适配器
- **步骤 10.1-10.9**: VASP Wiki 爬取、BM25/Qdrant 索引、证据包生成
- **步骤 11**: Agent 工具协议、LangGraph 编排、审批状态机
- **步骤 12**: VASP 输出解析（OSZICAR/OUTCAR/vasprun.xml）
- **步骤 13**: 真实集群执行运行时
- **步骤 14**: Skills 与 Reference_Demo 只读集成
- **步骤 15**: mCIF 工作流与强制人工审核
- **步骤 16**: 全组站点模板与每人身份合成

### 规划中（步骤 17+）

- Web 界面与项目管理
- 磁性分析工具集成（VASPKIT、电子结构可视化）
- 课题组协作功能（经验库、批量作业管理）

## 🔒 安全与隐私

本项目设计为**本地桌面工具**，不对外提供网络服务：

- 所有 API 密钥（LLM API Key）通过环境变量配置，不写入配置文件
- 集群 SSH 凭证存储在 `.gitignore` 保护的 `config/execution/cluster.json`
- POTCAR 文件不提交到 Git（VASP 版权限制）
- 关键操作（集群执行、文件修改）需人工审批

## 📄 许可证

MIT License

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request。请确保：

1. 代码通过 `pytest` 测试
2. 遵循项目的代码风格（参考 `AGENTS.md`）
3. 敏感信息（API Key、SSH 密钥）不提交到仓库

## 📮 联系方式

- GitHub Issues: https://github.com/Hazardforiza/magnetic-vasp-agent/issues
- 开发者: [@Hazardforiza](https://github.com/Hazardforiza)
