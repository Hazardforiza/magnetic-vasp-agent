# Wannier 节点只读核验（2026-10-16）

本记录只包含通过 `192.168.164.203:7722` 进入 `mu01` 后，对白名单计算节点执行的固定检查。没有上传文件，没有在 `mu01` 或计算节点启动 VASP、MPI 或 Wannier90，也没有修改用户计算目录；版本核验脚本曾在远端 `/tmp` 创建并随后删除短生命周期的探针日志。

## 节点和运行库

`cu20`、`cu23`、`cu39` 的瞬时 probe 没有发现高 CPU 的其他用户计算，分别观察到 40、48、96 个逻辑 CPU。该观察不等于资源分配，不能直接作为提交批准。`cu21`、`cu38`、`cu40`、`cu41`、`cu43`、`cu48`、`cu49`、`cu50` 在同一轮观察中有明显占用，不作为本轮候选。

登记的 `vasp/5.4.4-intel-wannier90` module 名称在 `cu20/cu23/cu39` 均加载失败；节点提供的是直接文件路径和 Intel oneAPI 运行库：

- `/public/vasp/vasp.5.4.4/bin/vasp_ncl`，24,944,608 bytes，SHA256 `e153e623411e61b3c8363e1d2ce78a00d9c5a896d2549ab9413627cbebf70ca4`。
- `/public/wannier90/3.1.0/bin/wannier90.x`，3,488,272 bytes，SHA256 `bc9fa42971e6d92015c488b1189f1fd2532e3dd299bdda9181c1fd4c2f4c44ca`。
- `/public/intel/oneapi/setvars.sh --force` 设置 MKL 运行库；不加载它时 `wannier90.x` 会因缺少 `libmkl_core.so.1` 退出 127。

在 `cu20` 加载 `setvars.sh` 后，`wannier90.x --version` 返回 `Wannier90: 3.1.0`。`cu23/cu39` 的文件哈希与 `cu20` 一致，但本次版本命令没有完成，因此只记录为共享文件证据，不能宣称三台都通过运行时版本检查。

## VASP 接口构建证据

`/public/vasp/source_code/vasp.5.4.4-intel-wannier90/makefile.include` 包含：

```text
CPP_OPTIONS+=-DVASP2WANNIER90v2
LLIBS+=/public/wannier90/3.1.0/lib/libwannier.a
```

`vasp_ncl` 的字符串和源码包含 `LWANNIER90`、`LWANNIER90_RUN`、`MLWF_WANNIER90`。源码 `mlwf.F` 还确认：

- `LWANNIER90` 和 `LWANNIER90_RUN` 是独立开关；
- 非 `LWANNIER90_RUN` 路径默认计算并写出 `.mmn`、`.amn`、`.eig`；
- `LWANNIER90_RUN` 会强制计算这些矩阵但走 Wannier90 library mode；
- `wannier90.win` 至少需要 `num_wann`；缺失时 VASP 会生成只含默认 `num_wann=NBANDS` 的文件，正式计算不能依赖这个自动值；
- `LWRITE_WANPROJ` 会强制 `LWANNIER90_RUN` 和 downsampling，不能在第一轮导出中无依据打开。

因此用户参考脚本里的 `ICHARG=11`、`NCORE=1`、`LWANNIER90=.TRUE.` 可以作为接口导出起点，但不能代表完整 Wannier 流程，也不应把 `LWANNIER90_RUN` 自动加进去。

## 当前阶段判断

状态为 `PRELIMINARY`：节点上有匹配的 VASP-Wannier90 构建和 W90 3.1.0，且 `cu20` 已通过不写计算文件的版本检查；尚未有真实 VASP 接口输出、`.win/.mmn/.amn/.eig` 维度检查、Wannier90 局域化或插值验收。

现有 FeO SCF 是共线 `10x8x5` Gamma 网格、`ENCUT=600`、Fe `U=5`，并已有受控回传的 `CHGCAR`。它不能直接作为 Wannier 结果；需要新建独立 run，使用全布里渊区均匀网格，重新明确 `NBANDS`、`num_wann`、投影、`dis_win/dis_froz` 和能量基准。Band 的 line-mode KPOINTS 不能替代这个接口网格。

后续顺序应为：

1. 在具名审批和资源 grant 后，以 `cu20` 或重新核验的空闲节点创建新的接口 run；固定加载 `setvars.sh`，保留 `LWANNIER90=.TRUE.` 的非库模式。
2. 受控回传并检查 `.win/.mmn/.amn/.eig` 的 header、band/k-point/projection 数和邻接关系。
3. 在同一 W90 3.1.0 环境运行 `wannier90.x -pp seed` 与局域化，检查 `wout/chk/hr`。
4. 用独立 k 点做 VASP/Wannier 能带误差验证；通过后才考虑 WannierBerri 响应。
