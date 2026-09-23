# VASP Wiki 受控抓取与切块

第 10.2 步通过 VASP Wiki 的 MediaWiki API 抓取 `config/knowledge/sources.json` 中预先登记的 official 页面。运行时快照和正文片段只保存在 Git 忽略的 `var/knowledge/`，仓库不提交官方正文。

## 安全与可追溯契约

- 只接受 HTTPS，主机固定为 `vasp.at` 或 `www.vasp.at`，页面路径必须是 `/wiki/index.php/<page>`。
- 不提供任意 URL、请求头、代理、shell 或页面脚本入口；跨域重定向、凭据、自定义端口、查询参数和片段均拒绝。
- 实际请求固定使用 `/wiki/api.php?action=parse`，单页响应上限 5 MiB，超时 20 秒，并在页面之间保留短暂间隔。
- 输出目录必须不存在。九个页面全部下载、解析和切块成功后才通过目录重命名发布；失败时不发布半成品。
- 每页保存原始 API JSON、page ID、revision ID、抓取时间、源 URL、API URL、响应大小和 SHA256。
- 每个 chunk 保存来源 ID、权威级别、任务/磁性模式、页面/revision、章节路径、正文 SHA256 和原始快照 SHA256。
- HTML 抽取器移除脚本、样式和 MediaWiki 编辑控件；网页内容始终作为不可信资料数据，不能变成工具指令。
- `POTCAR` 页面只抓取公开 Wiki 指南，不读取、生成或上传任何授权 POTCAR 文件。

## 命令

```powershell
uv run --frozen magvasp knowledge capture-wiki `
  --catalog config\knowledge\sources.json `
  --output var\knowledge\vasp-wiki-20260908-r1
```

命令没有覆盖开关。同一路径已存在时必须停止并选择新的 capture ID，以保留审计历史。

以后使用该 capture 前重新运行完整性检查：

```powershell
uv run --frozen magvasp knowledge verify-wiki `
  --catalog config\knowledge\sources.json `
  --capture var\knowledge\vasp-wiki-20260908-r1
```

检查器拒绝额外文件、缺失文件、符号链接、目录逃逸、catalog 哈希变化、响应/revision 元数据变化、chunk 顺序或来源变化，以及任一文件或正文 SHA256 不匹配。

## 2026-09-08 抓取结果

| 页面 | Revision | Chunks |
| --- | ---: | ---: |
| MAGMOM | 36612 | 5 |
| LNONCOLLINEAR | 36602 | 3 |
| LSORBIT | 27887 | 9 |
| ICHARG | 37119 | 4 |
| LDAU | 36499 | 2 |
| LMAXMIX | 27290 | 2 |
| KPOINTS | 35974 | 24 |
| ISMEAR | 37501 | 4 |
| POTCAR | 33125 | 5 |

合计 9 个页面、58 个 chunk。重新读取验证时，全部原始响应与 chunks 文件的 SHA256 和 manifest 一致；最大 chunk 长度为 1773 字符；chunk ID 无重复；抽取结果没有 MediaWiki `edit source` 控件或脚本文本。

运行时入口为 `var/knowledge/vasp-wiki-20260908-r1/manifest.json`。顶层 manifest 的 `catalog_sha256` 将本次快照绑定到抓取时的来源配置；每次更新应创建新目录，不能覆盖旧 revision。

## 当前限制

- 抓取的是页面当前 revision，不是针对 VASP 5.4.4 冻结的官方手册版本。所有 chunk 仍为 `source_validated=false`，版本适用性待人工核对。
- 当前切块依据 HTML 章节和块级语义，尚未建立参数别名、交叉链接图、公式结构或 VASP 版本标签。
- 尚未处理本地 DOC/DOCX/PDF 正文，也未执行 OCR。
- Wiki capture 已被 BM25 和本地 Qdrant 索引消费；索引只保存受校验文档的 provenance。当前尚未接入 LLM 问答，也未把授权 POTCAR 内容纳入索引。
- 58 个 chunk 只表示可检索语料已经形成，不代表 20 个基线问题已经通过。
