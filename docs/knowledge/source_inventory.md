# VASP 知识来源清单说明

第 10.1 步的机器可读来源声明位于 `config/knowledge/sources.json`。清单命令只读取文件元数据和字节以计算 SHA256，不解析 Word/PDF/JSON/Markdown 正文，不构建向量索引，也不连接网络或模型 API。

## 当前来源

- VASP Wiki 的 MAGMOM、LNONCOLLINEAR、LSORBIT、ICHARG、LDAU、LMAXMIX、KPOINTS、ISMEAR 和 POTCAR 页面在目录中登记为 `official/reference_only`。第 10.2 步已把当前 revision 抓取到 Git 忽略的运行时目录，但 VASP 5.4.4 适用性尚未人工验证；目录声明和运行时快照是两层记录。
- `vasp笔记/` 中的 Word 和 PDF 登记为 `personal_note/local_only`。原始目录保持未跟踪，不复制进 Git。
- `docs/vasp_notes_knowledge.md` 是笔记派生摘要，仍属于未完全验证的个人资料。
- `server_maintenance.html` 因包含明文凭据和管理员命令而永久排除；仓库只保存 `docs/knowledge/cluster_manual_sanitized.md` 中的脱敏审查与冲突说明，分类为未验证的个人资料。
- 集群策略文档和机器配置属于用户确认的 `team_policy`。
- 历史 INCAR/KPOINTS fixture 只证明解析和往返行为，当前标记为未验证的个人资料，不作为收敛或最佳参数证据。

没有任何现有历史案例被标记为 `validated_case`。只有在用户确认计算成功，并补齐 VASP 版本、POTCAR 标识、收敛证据和结果摘要后，才允许晋升。

2026-09-08 在当前电脑上完成只读验证并加入集群手册脱敏审查后：共声明 17 个来源，识别 22 个本地文档（44,122,139 字节）、9 个尚未抓取的官方引用、0 个缺失的可选来源和 1 组重复内容。重复文件为：

```text
vasp笔记/vasp课程ppt/vasp课程ppt/Lecture10.pdf
vasp笔记/vasp课程ppt/vasp课程ppt/lecture12_band structure.pdf
SHA256: a55eccedcd5de3121321e1690e28d5c0fd24d495f6d6aa4e46865f80b213c4a7
```

该统计是按当时的源文件字节计算的审计快照，不是正文已解析或知识已验证的声明。以后文件变化会产生新的哈希和统计结果。

## 生成清单

```powershell
uv run --frozen magvasp knowledge inventory `
  --catalog config\knowledge\sources.json `
  --output var\knowledge\source_inventory.json
```

输出文件包含来源声明哈希、每个本地文件的相对路径/大小/SHA256、缺失的可选来源、仅引用的外部来源和重复内容组。命令拒绝覆盖已有清单；需要重建时应选择新的输出路径或由用户明确处理旧报告。

## 尚未进行

- 未把 VASP Wiki 当前页面逐项核验为适用于 VASP 5.4.4；
- 未提取 DOC/DOCX/PDF 正文或 OCR；
- 未安装 Qdrant、embedding 或 reranker；
- 未把资料发送给任何外部 API；
- 未根据个人笔记自动修改 VASP 输入；
- 未把运行结果写回长期知识库。
