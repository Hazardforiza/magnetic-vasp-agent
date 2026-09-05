# Magnetic VASP Agent

面向磁性材料 VASP 计算的本地控制程序。最终在本机理解计算目标、生成经过校验的输入文件，再通过 SSH 进入集群并在获批的计算节点上直接运行 MPI。大模型 API 和 Agent 服务运行在本地，集群只需要 SSH、Intel MPI 与 VASP；本项目不假设或调用 Slurm。

## 当前进度

已实现本地授权赝势库扫描、POSCAR 类型模型、赝势选择与拼接、类型化 INCAR/KPOINTS、第六步工作流装配，以及第七步的 mCIF 磁性位点展开、磁矩坐标转换和候选结构映射。目前尚未接入大模型或集群提交，也没有执行真实 VASP 计算。

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
- 磁矩必须已按 POSCAR 原子顺序排列。三维分量属于 `SAXIS` 定义的 VASP 自旋坐标系，不自动等于晶体笛卡尔磁矩。第七步 mCIF 入口负责坐标转换和位点映射。
- `hubbard` 按元素给出 `l`、`u`、`j`，单位 eV，程序按物种顺序生成 `LDAUTYPE=2` 数组。SrRuO3 的 U=1.2、J=0.6 对应 U_eff=0.6 eV。U/J 不从 POTCAR 推断。
- ENCUT 下限取所选势的最大 ENMAX，不取整个势库最大值。满足下限不代表收敛。
- SOC/非共线使用正值或省略 ISYM 会提示复核；关闭对称性是项目策略，不是 VASP 的普遍硬性限制。
- 历史 `SOCSCALE` 保留并提示核实目标二进制支持情况，单文件 INCAR 预设不生成它；第六步工作流可通过受能力声明约束的选项生成。

INCAR 模块只声明重启依赖；第六步工作流层检查来源步骤及输入级兼容性；第七步 mCIF 入口负责磁矩和位点映射。实际 CHGCAR/WAVECAR 的内容、上游收敛和真实 VASP 执行仍需后续模块处理。`valid=true` 仅表示已实现的检查通过，不代表计算已经就绪或收敛。

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

## 第六步：本地计算与工作流装配

`workflow_models.py` 定义 `CalculationStep`、`Workflow`、声明式产物依赖和扫描模板。`workflow.py` 复用前四类生成器，完成整包交叉校验和本地文件装配。一个步骤对应一个目录；只有一个步骤的 Workflow 就是单次计算请求，不额外维护第二套装配接口。

### 一份请求生成整包

```powershell
uv run --frozen magvasp workflow validate `
  --request config\workflows\Mn3Al.scf-dos-band.json `
  --registry var\potcar_registry.json --policy config\potential_policy.json

uv run --frozen magvasp workflow build `
  --request config\workflows\Mn3Al.scf-dos-band.json `
  --registry var\potcar_registry.json --policy config\potential_policy.json `
  --output var\demo-workflow\Mn3Al

uv run --frozen magvasp workflow inspect var\demo-workflow\Mn3Al
```

`validate` 在内存中准备输入并核实所选 POTCAR 源哈希，不写输出。`build` 是本地 dry-run 装配：会写输入文件，但绝不执行 VASP、SSH、MPI、Slurm 或 LLM。`inspect` 检查已经生成的全部输入及每步 manifest 哈希；它不是科学有效性复核，也不能抵御输入和 manifest 被同时重写。命令成功退出 0；请求、依赖、完整性或文件错误退出 2。

输出布局如下，SCF 分别向 DOS 与 band 提供 CHGCAR，而不是 DOS 向 band 传递：

```text
Mn3Al/
  workflow.manifest.json
  scf/    POSCAR  POTCAR  INCAR  KPOINTS  manifest.json
  dos/    POSCAR  POTCAR  INCAR  KPOINTS  manifest.json
  band/   POSCAR  POTCAR  INCAR  KPOINTS  manifest.json
```

请求中的 `poscar` 相对路径以请求 JSON 所在目录为基准，不依赖命令的当前目录。每步 `id` 限于安全 ASCII 名称，Windows 大小写碰撞和保留设备名也会拒绝。`candidate_id` 用于结构来源标识，不用作输出路径。

装配入口要求单个标量缩放因子、明确的 Direct/Cartesian 坐标模式、有限且非奇异的晶胞。不支持三方向缩放或未建模的尾部数据；历史 Mn3Al 的全零速度块保留，非零速度输入拒绝。不会把不支持的 POSCAR 写法静默改写成另一种结构。

每步嵌入现有 `incar`、`kpoints` 请求；可选 `potential_plan` 接受人工审阅的选择结果，否则使用现有策略选择。三者的 goal 必须在本步内一致，但不同步骤可以使用不同 task。默认 XC 仍是 PBE.54；原子顺序不改变，不标准化晶胞。原子逐项索引、物种与 MAGMOM 切片记录于每步 manifest，不声称能从任意磁矩数组反推出原始位点身份。

`incar_options` 只接受明确的类型化选项：`ISTART`、`ICHARG`、`LWAVE`、`LCHARG`、`NBANDS`、`ISMEAR`、`SIGMA`、`LMAXMIX`、`SOCSCALE`。它不能覆盖请求中的 ENCUT、磁矩、磁性模式或 DFT+U 设置。覆盖后重新运行完整校验。工作流第一版限制 `ISTART=0/1`、`ICHARG=0/1/2/11`，不接受未建模的重启方式。

### 依赖契约与就绪状态

例如能带步骤声明：

```json
"dependencies": [{"artifact": "CHGCAR", "from_step": "scf"}]
```

- 必须为 INCAR 实际需要的每个产物提供且只提供一个来源。不接受缺少来源、额外无用依赖、未知步骤和循环依赖。输入步骤可以乱序列出，输出 manifest 记录拓扑执行顺序。
- 来源必须是不移动离子的自洽步骤，并明确设置 `LCHARG=True` 或 `LWAVE=True`。第六步不支持直接把尚未产生的弛豫后 CONTCAR 作为下游 POSCAR；需在真实弛豫后重新规划。
- 共享产物要求相同候选编号、相同物种/原子顺序、相同物理晶胞与位置，以及相同 POTCAR 内容哈希。位置比较允许整数周期平移，不允许位点重排；不会只比较化学式或晶格。
- 第一版保守要求 ENCUT、PREC、磁矩/自旋基底、SOC、DFT+U、LMAXMIX 等设置一致。相关壳层所需的 LMAXMIX 必须在上游足够，不能只在下游补大数值。这些是项目的输入级限制，不是所有 VASP 重启方式的普遍定律，也不足以证明实际二进制产物兼容。
- CHGCAR 允许不同 k 点采样，例如 SCF 网格到 band 路径或更密 DOS 网格。WAVECAR 则另外要求显式相同 NBANDS、相同 KPOINTS 和 ISYM；不把 SCF WAVECAR 自动交给不同采样的能带计算。
- 没有前置依赖的步骤标记 `prepared`，仅表示输入已准备。有依赖的标记 `waiting_for_upstream`，每条依赖为 `pending_upstream`，产物 SHA256 为 `null`。不会写空 CHGCAR/WAVECAR 占位，也不会因为用户自行放入一个同名文件就自动解除等待。

所有步骤始终为 `runtime_verified=false`。后续执行层必须检查目标 VASP、上游实际收敛及产物兼容性，记录真实文件哈希，再把产物送到下游。当前整包是可搬运的输入计划，不是已经可以无条件顺序启动的完整运行环境；没有生成 Slurm 脚本。

### 扫描模板

第六步在 `config/workflows/` 提供以下五份配置；第七步另外提供 mCIF 示例：

- `Mn3Al.scf-dos-band.json`：18×18×18 SCF、24×24×24 DOS 和历史六段能带路径，仍保留 U/K 断点。DOS 网格是待收敛测试的基线，不是历史精度保证。
- `CuMnAs.soc.json`、`SrRuO3.soc-u.json`：单步 SOC 和 SOC+U 整包。
- `Mn3Al.mae.json`：`001/011/111` 三方向独立自洽 SOC，生成 `mae-001/mae-011/mae-111`。各方向结构、赝势、k 点和其余 INCAR 严格相同，只变化 SAXIS。
- `CuMnAs.socscale.json`：0、0.5、1 的独立 SOCSCALE 扫描。示例故意设置 `capabilities.socscale=false`，默认拒绝生成；在确认目标 VASP 版本/补丁支持后才可开启，并用 `capabilities.evidence` 写明证据。测试中使用的虚拟能力声明不构成集群支持证据。

MAE 模板要求三维 MAGMOM 在 VASP 自旋坐标中只有 z 分量，且并非全零；因此改变 SAXIS 才能明确表示统一旋转磁矩方向。SAXIS 是笛卡尔方向，不自动等同于任意晶格下的 Miller 指数。本版不自动转换一般非共线纹理，不生成磁矩约束，也不保证自洽后方向保持不变。后处理仍需核查最终磁矩方向和精度后才能比较能量；固定电荷密度/力定理 MAE 不在此模板范围内。

SOCSCALE 同样只允许扫描该参数，其他输入保持不变。两类扫描在顶层 manifest 中声明比较组和方法，但 `results_available=false`，不伪造能量、MAE 或候选排序。

### 写盘与可追溯性

所有步骤先完成校验和源赝势读取，再在输出父目录下的独立临时目录写入完整树。发布前重新校验哈希，最后在当前 Windows 环境通过目录重命名发布；已有输出拒绝覆盖。常规写盘失败会清理本次临时目录，不改动参考算例或其他计算。进程被强制终止或断电时可能留下 `.magvasp-stage-*` 暂存目录，但不会被当成最终输入包；不承诺断电级事务。

每步 manifest 记录 goal、展开后的参数、原始 POSCAR 哈希、赝势来源哈希、四文件哈希、逐原子磁矩映射和校验报告。顶层记录完整请求、配方版本、来源哈希、依赖图、执行顺序与每步 manifest 哈希，不写入尚未产生的计算结果哈希。同一请求、同一来源路径/内容和同一锁定环境可生成逐字节相同的包；没有时间戳或随机任务编号。重新编译请求需要原始结构和授权势库，完整已生成输入则可单独做完整性检查。

回归测试中的赝势均为明确标注的合成文本，不包含授权数据。真实 POTCAR 仅保留在 Git 忽略的本机生成目录。这里只验证输入与依赖，不替代 ENCUT/k 点收敛测试或 VASP 实测。

## 第七步：mCIF 候选结构入口

`mcif_models.py` 定义类型化 `CandidateStructure`、原始位点、展开位点、来源操作和磁矩转换记录。`mcif.py` 使用 pymatgen 的 CIF 分词/解析、晶格、磁对称操作与 `Magmom` 坐标变换，增加严格输入边界和逐位点一致性校验。

### 查看候选与生成计算包

```powershell
uv run --frozen magvasp mcif show mcif\candidate_01_FeO_BNS2_7.mcif

uv run --frozen magvasp workflow build `
  --request config\workflows\FeO.generated.mcif.json `
  --registry var\potcar_registry.json --policy config\potential_policy.json `
  --output var\demo-mcif\FeO

uv run --frozen magvasp workflow inspect var\demo-mcif\FeO
```

`mcif show` 只读，输出标准化候选 JSON，不修改原文件；`--candidate-id` 可显式指定候选编号。退出码 0 为成功，2 为解析或一致性错误。工作流仍输出四个输入文件和 manifest，新增候选记录直接嵌入每步 manifest，不改变整包哈希检查的文件契约。

第七步示例引用已固化的真实 fixture，可在另一台机器检出后使用；同样支持把请求中的 `mcif` 指向原始文件。请求只能指定 `poscar` 或 `mcif` 之一。POSCAR 请求保持原有显式 `magmom` 用法；mCIF 请求在 `incar` 中必须指定 `moment_source="mcif"`，禁止同时提供另一份手写磁矩，避免两个来源相互覆盖。

示例使用 `ENCUT=600`、倒空间间距 0.25 Å⁻¹、PBE、无 U、无 SOC 的 spinor 静态计算，目的只是验证文件接通和方向保留。三个候选本身仍是共线 AFM；使用 `magnetic_mode="noncollinear"` 表示 VASP 的三维自旋表示，并不把材料重新分类为非共线磁序。Fe 采用当前策略的精确元素回退势 Fe，O 采用现有 O 策略。该配置不构成对 FeO 赝势、强关联处理或收敛精度的推荐；正式计算前需明确选择 Fe/Fe_pv、U/J、弛豫流程和精度测试。

### 真实样本和映射

`tests/fixtures/generated_mcif/` 保存用户生成的三个 FeO mCIF，逐字节哈希与来源一致；`provenance.json` 保存权重哈希、采样索引、固定轨道条件和参考磁矩。这些是 CrystalFormer 的条件生成候选，不是实验晶体。三者均为 BNS 2.7、非正交晶胞，原文两行位点展开为 2 Fe + 2 O，两个 Fe 磁矩相反，O 为零。

- 优先使用显式磁性操作和可选磁性平移中心操作，不用普通 CIF 的空间群循环代替磁性展开。纯时间反演和平移时间反演均按磁矩变号处理。
- `crystalaxis` 分量以单位晶轴为基底，先转换到同一晶胞的笛卡尔系。分数坐标旋转矩阵通过晶格基变换到笛卡尔系，再由 `MagSymmOp` 按轴矢量规则作用：空间反演不单独反转磁矩，时间反演会反转磁矩。不能对一般非正交晶胞直接把分数坐标旋转矩阵乘在晶轴磁矩分量上。
- 不求原胞、不约化磁胞、不旋转或标准化输入晶格。mCIF 的六个晶格参数通过 `Lattice.from_parameters` 建立确定的笛卡尔表示；此表示与原始晶轴分量一起记录，不能与另一个晶格朝向的笛卡尔数值混用。
- 物种按源文件首次出现顺序分组；同物种内部保持源位点/操作的展开顺序。坐标与磁矩同步移动。每个输出原子记录所有产生它的源标签、1 起始源行号、磁操作编号和中心操作编号。
- 同一位置重复出现时，物种和磁矩都一致才合并；即使输入已经列出完整磁胞也不会重复计数。等价位点磁矩冲突、混合物种重叠均报错，不自动平均或投影修复。
- 缺少占位率时记录 `OCCUPANCY_DEFAULT_ONE` 并采用 1；缺少 Wyckoff 标签时保存 `null`，不从文件名或生成轨道编号推造字母标签。已有 Wyckoff 标签仅作为来源声明保留。

生产解析会验证显式操作的单位元、闭合性、唯一性、晶格度量及位点磁矩一致性；BNS 编号保存在 `declared_bns`，不将其冒充为自动识别结果。三个真实 fixture 的测试另外使用 spglib 独立识别为 BNS 2.7，并与 pymatgen 的完整磁性解析对照。一般输入的声明 BNS 与完整磁群身份自动核验仍未实现。

### 两种磁矩输出方式

`moment_mode="preserve_cartesian"` 是默认方式，仅用于 `noncollinear` 或 `soc` 计算。将笛卡尔磁矩转换为请求 `saxis` 的自旋分量，输出 3N 个 MAGMOM 值，并做逆变换检查。SAXIS 先安全归一化，避免数值上溢/下溢，实际输出轴记录在转换报告中。改变 SAXIS 方向时分量会随之改变，实际初始磁矩方向保持不变。

`moment_mode="collinear_projection"` 必须显式选择，仅用于无 SOC 的 `collinear` 计算，且保持默认 SAXIS。程序先验证全部非零磁矩共线，以最大模长磁矩确定轴，并把其最大绝对分量定为正向，随后输出沿该轴的 N 个带符号投影值。相当于无 SOC 下把整个共线磁纹理全局旋转到计算自旋轴，报告明确标记 `orientation_preserved=false` 并保存原轴。不是只取晶轴 z 分量，也不对真正非共线结构丢弃横向分量；后者会直接报错。

第一份候选的共线路线约为 `MAGMOM=-1.41777418 1.41777418 0 0`；三维路线保留其实际笛卡尔方向。零磁矩不代表后续 VASP 中的磁矩约束，所有 MAGMOM 均为初始猜测。

不允许把 mCIF 来源直接套入第六步 MAE/SOCSCALE 扫描模板。尤其 MAE 的“只改变 SAXIS”在此入口会被坐标转换抵消，不能当作物理方向扫描。需要先新增显式磁矩旋转与相应比较契约；原有显式磁矩 POSCAR 扫描不受影响。

### 当前支持边界

- 只接受一个 CIF 数据块、显式三维磁性操作、明确纯元素、全部源位点的三维晶轴磁矩，零磁矩也必须明确提供。支持磁矩单独成表并按标签匹配，以及点号/下划线字段别名、标准不确定度、科学计数法和 CRLF。
- 不从 BNS 名称查询并猜测缺失的操作，不执行任何变换字符串代码。磁操作使用受限线性坐标表达式，再交给 pymatgen 解析；非法表达式、重复字段/别名、残缺循环、未知数值 `?`/`.`、非有限值、无效晶胞会报错。
- 部分占位、占位未知、混合物种、非公度/调制结构、不完整磁矩、非平移中心操作暂不支持。需要先进行有物理依据的结构枚举或格式扩展，不静默修补。
- 位点合并容差为 1e-5 Å；磁矩冲突、共线判断和坐标往返使用 1e-6 μB 绝对残差。输入大小上限 5 MB，最多 384 个合成磁操作、4096 个展开位点。数值容差检查不代替材料稳定性、磁基态或几何合理性筛选。
- 每步 manifest 保存原 mCIF 哈希、类型化候选、逐原子映射、磁矩转换方式和实际展开后的 INCAR 请求。原有 CHGCAR/WAVECAR 依赖契约继续适用，运行状态仍是 `runtime_verified=false`。

## 第八步：SSH 与直接 MPI 执行准备（默认禁用）

第八步适配本组的真实集群约束，不使用 Slurm：本机 Agent 通过 SSH 进入 `zyy@192.168.164.203`，该入口 `mu01` 只作为跳板，绝不在其上启动 VASP。允许的计算节点严格限定为 `cu21`、`cu22`、`cu23`、`cu41`、`cu42`、`cu43`、`cu48`、`cu49`、`cu50`。默认配置见 `config/execution/cluster.json`，其中 `execution_enabled` 永远为 `false`。

本步只实现三项确定性能力：

- `execution probe` 生成或执行固定的只读预检，核实主机、用户、CPU 拓扑、进程、负载、内存、VASP 模块、Intel MPI 和三个 VASP 可执行文件。预检不上传文件、不发送信号、不启动 MPI；`--connect` 也只有这一组固定命令。
- `execution plan` 重新检查工作流包的四个输入、逐项哈希、物种/赝势顺序、INCAR/KPOINTS/goal 和依赖状态，输出候选 `mpirun` 命令、环境变量、CPU 绑定和阻塞原因。它不读取 POTCAR 内容，也不把 `CHGCAR` 文件名当作已验证产物。
- `execution simulate` 只运行内置的极小 Python 子进程，测试完成、失败、超时、取消、日志和拒绝覆盖；不接受任意命令，不接触 VASP 或 SSH。

命令示例：

```powershell
uv run --frozen magvasp execution probe `
  --profile config\execution\cluster.json --node cu21

uv run --frozen magvasp execution plan `
  --bundle var\step7\FeO-generated `
  --profile config\execution\cluster.json `
  --request config\execution\FeO.cu21.preview.json

uv run --frozen magvasp execution simulate `
  --output var\execution-sim\complete --scenario complete
```

即使预检显示空闲，当前计划仍会阻塞：空闲快照不是资源分配，`PSR` 也不是 CPU 预留；没有验证 Intel MPI 的单节点启动和绑定行为前，不生成可执行提交脚本。cu21 当前已有其他用户的计算占用全部物理核，因此不能作为本轮测试目标，且不会被本程序干预。

第八步的 `ResourceAssessment.ready_to_run` 和执行报告的 `ready_to_run` 固定为 `false`。后续真正启用需要先由用户确认本组资源使用规则和空闲窗口，再用一个无科学结果的最小 MPI smoke test 验证单节点隔离、CPU 绑定、退出码和日志；这不是由 LLM 自主决定的步骤。LLM 可以提出执行计划和解释异常，但不能生成任意 shell、绕过节点白名单、选择 mu01、取消他人进程或自动重试。

## 第九步：受控 SSH/SFTP 上传（默认只生成计划）

第九步把已经通过第六、七步校验的 workflow 包绑定到一个新的远程目录。`execution transfer` 默认只生成传输计划，列出允许上传的输入文件、SHA256、固定 SSH/SFTP 参数和远端哈希校验脚本。上传范围严格为每步的 `POSCAR/POTCAR/INCAR/KPOINTS/manifest.json` 以及顶层 `workflow.manifest.json`；不会上传 `OUTCAR`、`CHGCAR`、`WAVECAR`、`vasprun.xml` 或任意用户文件。

远端 `remote_root` 必须由你先确认是组内批准、已存在且有配额的专用目录，例如 `/home/zyy/magvasp/runs`。程序只创建其下的本次唯一叶目录，并使用 `test ! -e` 和 `test ! -L` 拒绝覆盖或跟随链接；失败目录不会自动删除，避免误删已有数据。传输完成后，固定的 `sha256sum` 输出必须与本地清单逐项一致，否则报告失败，不进入运行阶段。

```powershell
# 只生成计划，不连接集群
uv run --frozen magvasp execution transfer `
  --bundle var\step7\FeO-generated `
  --profile config\execution\cluster.json `
  --request config\execution\FeO.cu21.transfer.json
```

只有显式加入 `--connect` 才会执行受控传输；这不是 VASP 提交，也不会执行 MPI。由于当前配置的 `remote_root` 为 `null`，命令会安全地拒绝，直到你确认远程专用目录后再另行修改配置。第九步仍不会提供 `launch`、`cancel` 或 `retry` 命令。

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
- 后续集群访问使用 `execution` 中的固定 SSH/直接 MPI 计划，不执行模型任意拼接的 shell 命令；本项目不使用 Slurm。
