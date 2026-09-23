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

### VASPKIT 边界

- `postprocess.vaspkit` 提供三个固定快捷任务：303 `band_kpoints` → `KPATH.in`；211 `band_dat` → `KLABELS` 加 `REFORMATTED_BAND.dat` 或 `BAND.dat`（仍识别旧适配器的小写 `band.dat`）；111 `tdos_dat` → `TDOS.dat`。输出必须由本次调用新建或改写，并通过对应格式解析，旧文件不能冒充成功。
- `postprocess.vaspkit_catalog` 只读列出本地 VASPKIT 1.5.1 示例家族和当前适配器状态。
- `postprocess.vaspkit_task` 是受限的数字任务入口：只接受程序中存在的 task id 和有界单行菜单输入，并记录本次变更文件及哈希。结果明确标为 `not_validated_generic_entry`，不能据此声称科学验证。
- `native` 后端只运行与宿主兼容的本地程序；Windows 检测到 Linux ELF 会失败并提示使用 `wsl`。`wsl` 后端只使用固定的 `wslpath`、`uname` 和 `--exec` 参数，不经过 shell；它要求 Windows 已安装可用的 WSL 发行版。`remote_ssh` 当前明确禁用，不能绕开集群审批控制面。
- 每次调用记录 VASPKIT 可执行文件路径、格式、大小、SHA256、可检测版本及宿主/WSL 环境。版本无法从运行横幅或同包 `changelog` 确认时记录为空，不猜测版本。
- `postprocess.plot_dos` 与 `postprocess.plot_band` 分别严格读取上述 TDOS 和能带文本，用 Matplotlib 生成静态 PNG，并返回数据文件、`KLABELS`（能带）和图片的 SHA256。未知列布局、非有限数、坐标乱序或标签范围不匹配都会失败。

```powershell
# Windows + 已安装 WSL 发行版：从 DOSCAR 生成并校验 TDOS.dat
uv run --frozen magvasp postprocess vaspkit path\to\dos `
  --task tdos_dat --backend wsl `
  --executable path\to\vaspkit

# 从 EIGENVAL 等必需文件生成并校验能带文本
uv run --frozen magvasp postprocess vaspkit path\to\band `
  --task band_dat --backend wsl `
  --executable path\to\vaspkit

# 绘制静态图并输出带哈希的 JSON 清单
uv run --frozen magvasp postprocess plot-dos path\to\TDOS.dat `
  --output path\to\tdos.png
uv run --frozen magvasp postprocess plot-band path\to\REFORMATTED_BAND.dat `
  --labels path\to\KLABELS --output path\to\band.png
```

固定的 303/211/111 适配器现已具备文件级格式验证，但这仍不证明物理模型、收敛参数或科学结论正确。其他 DOS/PDOS、电荷、势能、磁性和声子任务仍需逐项增加版本化菜单参数、必需输入、预期输出与科学校验规则。

## 测试

离线单测使用从真实 CuMnAs 算例裁剪出的紧凑 fixture（`tests/fixtures/postprocess/{collinear,soc}`）。另有一组交叉校验测试仅在本机通过 `MAGVASP_REAL_ARCHIVE` 指定历史算例归档时运行、否则自动跳过，用真实 `OUTCAR`/`OSZICAR`/`vasprun.xml` 验证三源一致。
