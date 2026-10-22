# VASP 知识来源清单说明

机器可读来源声明位于 `config/knowledge/sources.json`。仓库只包含脱敏示例、通用项目文档和公开资料引用；组员的课程材料、个人文献、审核记录和本机运行经验应留在本机，不要把本机目录加入 Git 或外部索引。

清单命令读取登记来源的元数据并计算 SHA256，不解析 Word/PDF 正文、不构建向量索引，也不连接网络或模型 API。来源类型、验证状态和内容处理策略应与材料来源和授权范围一致：

- 官方资料默认登记为 `official/reference_only`；核验具体 VASP 版本前不得声称适用。
- `team_policy` 只应包含管理员确认且允许共享的通用运行规则，不得记录个人账号或站点凭据。
- `personal_note/local_only` 只在本机处理；发送到远程 embedding 或 LLM 前需另行审查。
- 测试 fixture 只证明程序处理行为，不证明参数科学有效或结果收敛。
- 没有完整输入、版本、赝势标识和收敛证据的案例不得标记为 `validated_case`。

发布版的 `release_samples/` 仅用于验证目录和检索流程，不代表用户的真实文献、课程或技能库。

## 生成清单

```powershell
uv run --frozen magvasp knowledge inventory `
  --catalog config\knowledge\sources.json `
  --output var\knowledge\source_inventory.json
```

输出文件包含来源声明哈希、每个本地文件的相对路径/大小/SHA256、缺失的可选来源、仅引用的外部来源和重复内容组。不要将包含个人文件路径的 inventory 提交到 Git。

## 尚未进行

- 未把 VASP Wiki 当前页面逐项核验为适用于任何特定 VASP 版本；
- 未提取 DOC/DOCX/PDF 正文或 OCR；
- 未安装 Qdrant、embedding 或 reranker；
- 未把资料发送给任何外部 API；
- 未根据本机个人笔记自动修改 VASP 输入；
- 未把运行结果写回长期知识库。
