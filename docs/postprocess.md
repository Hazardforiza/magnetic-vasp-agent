# VASP 输出后处理（第 12 步）

把一个算完的 VASP 目录解析成**类型化、带溯源哈希**的结果，是磁性材料"读结果、比磁序"的基础。解析器与本项目的输入解析器同风格：确定性、可完全离线测试、拒绝静默漂移。

## 设计原则

- **自研解析器**：与 INCAR/KPOINTS/POSCAR 一致，保证可审计与可控，不依赖第三方输出解析器的版本行为。
- **内存安全**：`vasprun.xml` 用流式 `xml.etree.ElementTree.iterparse` 逐元素解析并及时 `clear()`，因此几百 MB 的真实文件（本仓历史算例最大 ~655 MB）也不会撑爆内存。
- **只存摘要**：绝不保存整段 DOS / 本征值数组；DOS 只保留费米面态密度、按自旋分辨与粗略能隙。
- **溯源**：每个来源解析结果都带 `source_sha256`；合并结果对其规范化 JSON（排除 `result_sha256` 本身）计算 `result_sha256`。

## 数据来源与职责

| 文件 | 角色 | 提取内容 |
| --- | --- | --- |
| `OSZICAR` | 便宜的能量/磁矩来源 | SCF 迭代节奏、最终 `F=`/`E0=`/`d E`、共线净磁矩或非共线三分量磁矩 |
| `OUTCAR` | **主源（磁性关键量）** | `ISPIN`/`LNONCOLLINEAR`/`LSORBIT`、磁性模式、`reached required accuracy`、`TOTEN`/`energy(sigma->0)`、费米能、**逐离子局域磁矩**、最大受力 |
| `vasprun.xml` | 结构化交叉校验 | 版本、`ISPIN`/非共线/SOC、电子/离子收敛、最终能量、费米能、DOS 摘要 |

### 磁性模式判定

- `LSORBIT=T` → `soc`
- `LNONCOLLINEAR=T` 且 `LSORBIT=F` → `noncollinear`
- `ISPIN=2` → `collinear`
- `ISPIN=1` → `nonmagnetic`

共线情形逐离子磁矩为带号标量（OUTCAR `magnetization (x)` 块的 `tot` 列）；非共线/SOC 情形从 `magnetization (x)/(y)/(z)` 三块组装为矢量，并给出模长。

### DOS 摘要

`vasprun.xml` 的 `total/array/set` 子集数量决定自旋结构：1 个为非磁总态密度，2 个为共线上/下自旋，4 个为非共线（总 + 磁化 x/y/z，取第 1 个为总态密度）。能隙仅在费米能落在总态密度近零窗口内时给出，否则判为金属（无能隙）。这是从 DOS 网格得到的粗略估计，不是严格的能隙计算。

## 合并结果 `CalculationResult`

- 能量优先取 OUTCAR `TOTEN`，缺失时回退 vasprun，再回退 OSZICAR。
- 若目录内有 `POSCAR`/`CONTCAR`，按 POSCAR 顺序为每个磁矩离子赋 species。
- 在多源之间做能量/磁矩/费米能一致性交叉校验（默认容差：能量 1e-2 eV、磁矩 5e-2 μB、费米能 1e-1 eV），不一致时加入告警。
- 截断或未完成的算例判为未收敛并给出告警，而不是崩溃。

## 命令

```powershell
# 解析目录，输出类型化 JSON；未收敛时退出码为 1
uv run --frozen magvasp postprocess parse path\to\calc_dir

# 人类可读摘要
uv run --frozen magvasp postprocess show path\to\calc_dir
```

## Agent 集成

后处理注册为只读 Agent 工具 `postprocess.parse`，经审批门自动批准并带哈希执行。LLM 因此可以"读结果并总结"，但仍不触碰集群、不执行 VASP。

## 测试

离线单测使用从真实 CuMnAs 算例裁剪出的紧凑 fixture（`tests/fixtures/postprocess/{collinear,soc}`）。另有一组交叉校验测试仅在本机存在历史算例归档（`vasp实例/CuMnAs_vasp`）时运行、否则自动跳过，用真实 `OUTCAR`/`OSZICAR`/`vasprun.xml` 验证三源一致。
