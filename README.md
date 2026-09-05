# Magnetic VASP Agent

面向磁性材料 VASP 计算的本地控制程序。最终在本机理解计算目标、生成经过校验的输入文件，再通过 SSH 向 Slurm 集群提交任务。大模型 API 和 Agent 服务运行在本地，集群只需要 SSH、Slurm 与 VASP。

## 当前进度

已实现本地授权赝势库扫描、POSCAR 类型模型、赝势选择与拼接，以及类型化 INCAR、KPOINTS 的解析、校验和生成。目前尚未接入 mCIF 转换、大模型或集群提交，也没有执行真实 VASP 计算。

## 本地环境

推荐使用 Python 3.11 和锁文件：

```powershell
cd C:\Users\18355\magnetic-vasp-agent
uv sync --extra dev --frozen
uv run --frozen magvasp doctor
```

`uv.lock` 固定依赖版本。不使用 uv 时也可安装开发环境，但下面的 pip 命令按版本范围解析依赖，不等同于锁文件安装：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\magvasp doctor
```

## 赝势库与 POTCAR

扫描本机 PBE.54 库：

```powershell
uv run --frozen magvasp potcar scan `
  --root "C:\Users\18355\vasp实例\vasp.pot\vasp.pot\赝势库\potpaw_PBE\potpaw_PBE.54" `
  --family PBE_54 --output var\potcar_registry.json
```

注册表只记录元数据、路径和 SHA256。它与生成的 POTCAR 都被 Git 忽略。POTCAR 内容来自本地授权库；模型未来负责提出选择方案，程序负责查验和拼接。

优先从 POSCAR 读取物种顺序并生成选择计划：

```powershell
uv run --frozen magvasp structure show tests\fixtures\Mn3Al\POSCAR

uv run --frozen magvasp potcar select `
  --registry var\potcar_registry.json --policy config\potential_policy.json `
  --poscar tests\fixtures\Mn3Al\POSCAR --candidate-id Mn3Al `
  --task static --magnetic-mode collinear --output var\Mn3Al.potential-plan.json

uv run --frozen magvasp potcar build `
  --registry var\potcar_registry.json --plan var\Mn3Al.potential-plan.json `
  --output var\example-potcar\Mn3Al\POTCAR
```

尚无 POSCAR 时可使用 `--element Mn --element Al`，但与 `--poscar` 互斥，且必须提供 `--candidate-id`。目前仅接受带显式物种行的 VASP 5 格式 POSCAR。

`tests/fixtures/*/POSCAR` 来自 Mn3Al、CuMnAs 与单斜 SrRuO3。现有赝势策略与全部 23 个历史算例所用势在归一化换行后逐字节一致。SrRuO3 历史 POTCAR 的换行符不同，所以原始整文件哈希不同。

规则选择器是可测试的基线。未来 LLM 必须输出相同的 `PotentialPlan` 并通过校验。历史算例不构成对所有计算目标的普适保证，策略置信度也不是经过统计校准的正确率。

## 第四步：类型化 INCAR

`incar_models.py` 中的 `IncarSpec` 定义标签和类型，拒绝未知标签、非有限数值和 JSON 字符串到布尔值的隐式转换。`incar.py` 在 pymatgen 基础上增加完整数值检查及行号诊断，支持：

- `!`、`#` 注释，分号赋值，反斜杠续行。
- `n*value` 缩写、科学计数法、Fortran D 指数及 CRLF。
- 无赋值的历史标题，记录 `IGNORED_TEXT` 诊断。
- 相同值重复赋值警告；冲突重复赋值报错。

历史 `.Ture.` 默认解释为 `.TRUE.`，记录 `LEGACY_BOOLEAN` 警告；`--strict` 拒绝该拼写。输出统一为 `.TRUE.` / `.FALSE.`。未知标签必须先扩展模型，不会被静默丢弃。

23 份历史文本固化在 `tests/fixtures/historical_incars.json`，保留原始文件哈希，fixture 文本统一为 LF。往返测试比较类型化参数，不要求保留注释和排版。

### 命令示例

```powershell
uv run --frozen magvasp incar show "C:\Users\18355\vasp实例\Mn3Al_nosoc\nosoc_scf\INCAR"

uv run --frozen magvasp incar validate "C:\Users\18355\vasp实例\Mn3Al_nosoc\nosoc_scf\INCAR" `
  --poscar tests\fixtures\Mn3Al\POSCAR `
  --plan var\Mn3Al.potential-plan.json --registry var\potcar_registry.json `
  --executable vasp_std

uv run --frozen magvasp incar build `
  --request config\incar_requests\Mn3Al.static.json `
  --poscar tests\fixtures\Mn3Al\POSCAR `
  --plan var\Mn3Al.potential-plan.json --registry var\potcar_registry.json `
  --output var\demo-incar\Mn3Al\INCAR
```

`show` 输出类型化参数和诊断。`validate` 检查磁矩长度、物种数组、所选势 ENMAX 和执行程序，列出重启依赖。退出码 0 表示无校验错误，1 表示一致性检查失败，2 表示解析或命令错误。警告不会使历史导入自动失败。

`build` 生成新 INCAR 和 `INCAR.manifest.json`，记录预设名称/版本、参数、校验结果以及输出/来源哈希，不覆盖已有输出。请求 goal 必须与赝势计划一致，切换任务或磁性模式时应重新生成计划。CuMnAs SOC 和 SrRuO3 SOC+U 示例也在 `config/incar_requests/`。

### 计算预设

预设是收敛测试起点，不是逐项复制历史参数。`static` 对应新启动 SCF。第 1 版要求明确提供 `encut`、`magmom`，使用 PBE.54、`PREC=Accurate`、`ISMEAR=0`、`SIGMA=0.05` 和 `ISYM=-1`。

| 任务 | 电荷与离子设置 | 其他行为 |
| --- | --- | --- |
| `static` | `ISTART=0`、`ICHARG=2`、`NSW=0` | 写出 CHGCAR |
| `relax` | 新启动，`NSW=200`、`IBRION=2`、`ISIF=2` | 只弛豫原子位置，`EDIFFG=-0.02` |
| `dos` | `ICHARG=11`、`NSW=0` | 依赖兼容 SCF CHGCAR，`NEDOS=2001` |
| `band` | `ICHARG=11`、`NSW=0` | 依赖兼容 SCF CHGCAR，`LORBIT=12` |
| `magnetic_anisotropy` | 单方向的新启动自洽 SOC | `EDIFF<=1e-8`、`LORBMOM=.TRUE.` |

- 共线模式需要 N 个带符号磁矩，使用 `ISPIN=2` 和 `vasp_std`。
- 非共线/SOC 需要 3N 个数值，开启 `LNONCOLLINEAR`，使用 `vasp_ncl`。新输入省略 ISPIN，避免冗余及部分新 VASP 版本的不兼容。
- 磁矩必须已按 POSCAR 原子顺序排列。三维分量属于 `SAXIS` 定义的 VASP 自旋坐标系，不自动等于晶体笛卡尔磁矩。未来 mCIF 入口负责坐标转换和位点映射。
- `hubbard` 按元素给出 `l`、`u`、`j`，单位 eV，程序按物种顺序生成 `LDAUTYPE=2` 数组。SrRuO3 的 U=1.2、J=0.6 对应 U_eff=0.6 eV。U/J 不从 POTCAR 推断。
- ENCUT 下限取所选势的最大 ENMAX，不取整个势库最大值。满足下限不代表收敛。
- SOC/非共线使用正值或省略 ISYM 会提示复核；关闭对称性是项目策略，不是 VASP 的普遍硬性限制。
- 历史 `SOCSCALE` 保留并提示核实目标二进制支持情况，当前预设不生成它。

CHGCAR/WAVECAR 兼容性、mCIF 转换、各向异性能量比较的一致性设置和真实 VASP 执行仍需后续模块处理。`valid=true` 仅表示已实现的检查通过，不代表存在完整可运行任务。

## 第五步：类型化 KPOINTS

`kpoints_models.py` 定义 `KpointsSpec`，`kpoints.py` 负责解析、规范化输出和跨文件校验，`kpoints_recipes.py` 根据共同计算目标生成采样。

真实盘点结果是 **14 份 Gamma 网格、9 份 Line-mode**。23 份文本及原始 SHA256 固化在 `tests/fixtures/historical_kpoints.json`。历史文件没有 Monkhorst–Pack 或显式加权列表，这两类使用 `tests/fixtures/kpoints_synthetic/` 中明确标注为合成数据的 fixture 验证。

### 格式与模型

| `mode` | 必需字段 | 支持内容 |
| --- | --- | --- |
| `gamma` | `mesh` | Gamma 网格和可选 `shift`，默认偏移为零 |
| `monkhorst` | `mesh` | Monkhorst–Pack 网格和可选 `shift` |
| `line` | `coordinate_mode`、`points_per_segment`、`segments` | 每段独立的起点、终点及标签 |
| `explicit` | `coordinate_mode`、`points` | 带权重的显式点，可附 `tetrahedra` 连接块 |

坐标模式为 `reciprocal` 或 `cartesian`。解析器支持正负号、E/D 指数、CRLF、`!`/`#` 注释，以及带或不带 `!` 的路径标签。未知格式、额外数据行、点数不符、不完整端点对、负权重和越界四面体索引都会报错。`Automatic` 单长度模式和自动基矢模式暂不支持。

每个 `LineSegment` 直接包含 `start` 与 `end`，端点对象各自保存坐标和标签，因此段与标签不会因平行数组错位。历史导入可保存无标签端点，但能带任务校验要求每个端点都有标签。

Mn3Al 的实际路径是 `GAMMA-X-U | K-GAMMA-L-W-X`，包含 **6 段**，`U` 与 `K` 之间没有路径段。`points_per_segment=40` 表示每段 40 个点，不是整个文件的端点数。程序不会连接断开的路径或改变磁性晶胞。

显式点保留浮点权重，例如 `0.25` 不会被写成 `0`；四面体连接采用 VASP 的 **1 起始索引**。规范化输出使用 UTF-8/LF，保留参数语义而非原始排版。

### 任务预设与一致性

- `static`（SCF）、`dos`、`relax`、`magnetic_anisotropy` 使用积分采样，拒绝 Line-mode。默认生成均匀网格；用户也可明确提供带正积分权重的显式列表。
- 按本项目约定，`band` 必须使用 Line-mode，不能用普通网格或显式加权列表替代。此约定是当前流程范围，不代表 VASP 不支持其他能带计算方式。
- 第 1 版能带生成接受用户给出的分段路径，不自动推导高对称路径。缺少路径时直接报错，不会默默标准化或约化磁性晶胞。
- KPOINTS 请求的 goal 必须与 `--incar-request` 中的 goal 一致，包括任务、磁性模式等。实际 INCAR 也会按同一 goal 重新检查，例如能带要求 `ICHARG=11`、静态任务不能移动离子、MAGMOM 必须匹配磁性模式。
- `ISMEAR=-5/-4` 拒绝 Line-mode 和少于 4 点的采样。自动网格满足最低点数后，报告标记为 `mesh_precheck_only`，仍需从 VASP/IBZKPT 验证实际约化点及四面体构建；网格总点数不是不可约点数。
- 显式列表使用四面体积分时必须提供连接块；模型校验连接行数、体积系数及索引范围。连接存在并不证明其完整覆盖布里渊区或计算已经收敛。

新 KPOINTS 校验会嵌套第四步的 INCAR 校验结果；此命令没有提供赝势上下文，因而其中可能出现 `ENCUT_UNCHECKED`。需要检查截断能时，仍使用带 `--plan/--registry` 的 `incar validate`。

### 网格与路径请求

网格可以明确指定，也可以使用倒空间间距：

```json
{
  "goal": {"task": "static", "magnetic_mode": "collinear", "xc_family": "PBE_54"},
  "spacing": 0.15,
  "centering": "gamma"
}
```

`spacing` 单位是 Å⁻¹，倒格矢包含 `2π`；各方向网格数为 `max(1, ceil(|b_i| / spacing))`。转换考虑 POSCAR 的正比例缩放和负值体积缩放，不改变原子顺序或晶胞。它只控制采样间距，不代表收敛保证。

`sampling` 与 `spacing` 必须二选一。`centering` 只用于 spacing 请求。显式路径的坐标属于实际计算晶胞；Cartesian 路径按 VASP 的坐标单位解释，不自动当作上述 spacing 的 Å⁻¹ 坐标。

现成请求位于 `config/kpoints_requests/`：

- `Mn3Al.static.json`：`18×18×18` Gamma 网格。
- `CuMnAs.soc.json`：`17×10×17` Gamma 网格。
- `SrRuO3.soc-u.json`：`11×11×8` Gamma 网格。
- `Mn3Al.spacing.json`：按倒空间间距生成网格。
- `Mn3Al.band.json`：保留原始断点的 6 段路径，与 `config/incar_requests/Mn3Al.band.json` 配套。

### 命令示例

```powershell
uv run --frozen magvasp kpoints show tests\fixtures\Mn3Al\KPOINTS

uv run --frozen magvasp kpoints validate tests\fixtures\Mn3Al\KPOINTS `
  --poscar tests\fixtures\Mn3Al\POSCAR `
  --incar var\demo-incar\Mn3Al\INCAR `
  --incar-request config\incar_requests\Mn3Al.static.json

uv run --frozen magvasp kpoints build `
  --request config\kpoints_requests\Mn3Al.static.json `
  --incar-request config\incar_requests\Mn3Al.static.json `
  --incar var\demo-incar\Mn3Al\INCAR `
  --poscar tests\fixtures\Mn3Al\POSCAR `
  --output var\demo-incar\Mn3Al\KPOINTS
```

上述 INCAR 来自第四步示例。能带示例需先以 `task=band` 生成新的赝势计划和 INCAR，再给 KPOINTS 命令提供对应能带请求，不应直接复用 SCF 的 goal。

`build` 同时写出 `KPOINTS.manifest.json`，记录共同 goal、预设版本、坐标约定、校验结果以及 KPOINTS、POSCAR、INCAR、两份请求文件的哈希。已有输出不覆盖。`show/validate/build` 的退出码约定与 INCAR 命令一致。

## 历史审计与测试

按相邻 POSCAR 和所选势检查全部原始 INCAR：

```powershell
uv run --frozen python scripts\audit_incar_references.py "C:\Users\18355\vasp实例" `
  --registry var\potcar_registry.json --policy config\potential_policy.json `
  --output var\my-incar-audit.json

uv run --frozen python scripts\audit_kpoints_references.py "C:\Users\18355\vasp实例" `
  --output var\my-kpoints-audit.json

uv run --frozen pytest
```

审计脚本记录参数和诊断，只写新报告，不修改参考输入，也不读取 POTCAR 内容。KPOINTS 审计用已知历史目录名称确定任务，再检查相邻 INCAR/POSCAR；这是算例审计约定，正式生成命令始终要求显式计算目标。两类原始文件均已完成 23/23 的解析、校验和往返检查，警告保留在报告中。

## 文件与执行边界

- POTCAR 内容不会发送给大模型或提交到 Git。
- 赝势选择只能引用注册表中存在的势，拼接前校验源文件哈希。
- `var/` 保存本机注册表、临时文件及报告，不进入 Git。
- 后续集群访问使用指定 SSH 配置及类型化 Slurm 操作，不执行模型任意拼接的 shell 命令。
