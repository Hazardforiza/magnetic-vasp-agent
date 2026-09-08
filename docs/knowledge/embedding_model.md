# 10.6 本地多语言 embedding 模型

本项目的第一条真实语义 embedding 评测模型采用 FastEmbed 的 ONNX 端口：

- 公开模型名：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- FastEmbed 实际仓库：`Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q`
- 固定 revision：`faf4aa4225822f3bc6376869cb1164e8e3feedd0`
- 主模型文件：`model_optimized.onnx`
- 维度：384；FastEmbed 0.8 使用 attention-mask mean pooling，输出默认不做 L2 归一化
- 语言/长度：约 50 种语言，最多 512 tokens
- 模型许可：Apache-2.0（固定 revision 的 README front matter 和 Hub API `cardData.license` 均为 `apache-2.0`）

文件和来源校验记录在运行时目录的 [`model-manifest.json`](../../var/knowledge/models/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/model-manifest.json)。清单记录每个文件的大小、SHA256、上游 Git blob 或 LFS SHA256，以及固定 revision 和许可证据。模型目录还保留固定 revision 的 `upstream-revision-metadata.json`，便于离线复核。本次直连 `huggingface.co` 超时，故从 `hf-mirror.com` 的相同固定 revision 下载；所有文件随后与 revision API 的文件大小以及 LFS SHA256 或 Git blob SHA1 逐项核对。

已下载的上游文件为 `config.json`、`model_optimized.onnx`、`ort_config.json`、`special_tokens_map.json`、`tokenizer.json`、`tokenizer_config.json`、`unigram.json`、`README.md` 和 `.gitattributes`。模型位于 `var/knowledge/models/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/`，不纳入 Git 提交；使用时必须指定该本地目录和 `local_files_only=True`，并设置 `HF_HUB_OFFLINE=1` 或等效的网络隔离。

2026-09-08 已在 CPUExecutionProvider 上完成离线可加载性 smoke test：先将 Python socket 连接函数改为直接抛错，再加载本地目录并编码一条英文和一条中文文本；两条结果均为有限的 384 维向量，跨语言余弦相似度为 0.84796。该测试只证明模型可离线加载，不代表 20 题检索评测已经达标；评测必须与 hash/BM25 基线使用同一套问题、过滤器和 provenance 检查。

上游 API 和模型卡的网络访问只用于一次性准备模型，未发送任何项目语料。运行时检索路径应拒绝自动下载或远程 API，并在模型文件缺失、哈希不符或 revision 不符时停止。
