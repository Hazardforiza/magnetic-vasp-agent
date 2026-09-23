# 10.6 embedding 评测报告

## 范围

本报告比较三种本地检索后端：BM25、`hash-token-bigram-v1` 工程基线，以及固定模型 revision 的 MiniLM semantic embedding。三者使用相同的 74 文档语料、20 个问题、过滤器、返回数量和 evidence provenance 检查。评测只统计检索覆盖，不判断回答是否科学正确，也不替代人工审核。

评测套件为 `tests/fixtures/knowledge/rag_evaluation_cases.json`。每题包含预期事实、证据词、禁止结论和要求的来源类型。`retrieval_rate` 表示至少返回一个文档；`required_source_type_coverage` 是每题要求来源类型的平均召回；`evidence_term_coverage` 是证据词命中比例；`forbidden_exact_quote_rate` 用于确认返回正文没有直接命中禁止引用。

## 结果

| 后端 | 索引 | 检索率 | 来源类型覆盖 | 证据词覆盖 | 禁止引用率 |
| --- | --- | ---: | ---: | ---: | ---: |
| BM25 | `bm25-20260908-r2` | 0.90 | 0.650 | 0.6806 | 0.0 |
| Hash | `qdrant-20260908-r2` | 0.95 | 0.725 | 0.7222 | 0.0 |
| MiniLM semantic | `minilm-20260908-r1` | 0.95 | 0.800 | 0.7639 | 0.0 |

MiniLM 与 hash 基线都返回了 20 题中的 19 题证据。MiniLM 的来源类型覆盖比 BM25 高 0.150，比 hash 高 0.075；证据词覆盖比 BM25 高约 0.0833，比 hash 高约 0.0417。仍有问题出现无证据、证据词缺失或只命中不适用来源的情况，因此不能据此自动生成或执行 VASP 输入。

## 模型 provenance

- 模型名：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- 上游 ONNX 仓库：`Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q`
- 固定 revision：`faf4aa4225822f3bc6376869cb1164e8e3feedd0`
- 许可证：Apache-2.0
- 向量维度：384，运行时做 L2 归一化
- 运行时：FastEmbed 0.8.0、ONNX Runtime 1.29.0、tokenizers 0.23.2
- 索引 manifest fingerprint：`8cde3d11b0e8c21d23b743b568b9b58bb3b7caca9cb1c98be45ce29ff8526605`

上游下载清单 `model-manifest.json` 记录固定 revision 的文件来源和逐文件哈希；运行时清单 `magvasp-model.json` 是程序使用的类型化契约。两者用途不同：前者用于一次性 provenance 审查，后者用于加载前完整性校验。模型目录位于 `var/knowledge/models/`，被 Git 忽略，运行时必须通过 `--model-dir` 指定；不会自动联网下载。

## 限制和后续

本评测没有验证 VASP 版本适用性、语义答案正确性或来源间冲突。Wiki chunk 仍需针对目标 VASP 版本人工确认，个人笔记和集群资料也按其 `validated` 状态处理。第 10.7 步应在证据包之上增加带 `[E#]` 引用的回答草稿、引用完整性校验和人工审核状态，并继续保持 `draft_only`。

原始机器可读报告：

- `var/knowledge/evaluation-10.6-bm25.json`
- `var/knowledge/evaluation-10.6-hash.json`
- `var/knowledge/evaluation-10.6-minilm.json`
