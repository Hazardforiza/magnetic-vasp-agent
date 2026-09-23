# 本机网页使用说明

网页是每位使用者在自己电脑上启动的 MagVASP 工作台，地址固定为本机回环接口。将源码放到 GitHub 只方便组员安装更新，不会自动提供可远程访问的在线服务。

## 首次准备

在仓库根目录使用 Python 3.11 和 `uv`：

```powershell
uv sync --extra agent --extra web --frozen
uv run --extra agent --extra web --frozen magvasp doctor
```

生成计算输入还需要本人获授权的 POTCAR 库。先扫描为本机注册表；网页只使用该注册表选择和核验赝势，不在审核页显示 POTCAR 正文：

```powershell
uv run --extra agent --extra web --frozen magvasp potcar scan `
  --root "C:\path\to\potpaw_PBE.54" `
  --family PBE_54 --output var\potcar_registry.json
```

使用自然语言 Agent 时，从 `config/agent/llm.example.json` 建立本机 `config/agent/llm.json`，配置模型服务，并在当前终端设置该配置所指向的 API key 环境变量。每位组员使用自己的模型凭据。集群执行还需按 [`member_setup.md`](knowledge/member_setup.md) 配置自己的 SSH 身份和站点模板；网页不会代管集群密码。

## 启动与使用

```powershell
uv run --extra agent --extra web --frozen magvasp web
```

默认打开 `http://127.0.0.1:8765`。浏览器没有自动出现时可手动打开该地址。端口占用时使用 `--port 8766`；不希望自动打开浏览器时加 `--no-browser`。按 `Ctrl+C` 停止本机服务。

- **项目**：上传一份 `.mcif` 或 `.cif`，选择 SCF 静态计算或结构弛豫，填写 ENCUT 与 k 点间距，可选择附加 DOS 步骤。生成后逐项查看 POSCAR、INCAR、KPOINTS、赝势名称和哈希；由具名审核人接受、退回或要求修改。接受输入只代表这一版输入通过人工审核，不是启动集群作业的授权。
- **Agent**：输入计算目标；已配置模型时创建本机会话。工具需要批准时，页面显示待审批调用，由具名审核人逐项决定后继续。实时探测、资源批准、smoke、传输、启动仍受原有分级审批和运行证据约束。
- **结果**：导入 `OSZICAR`、`OUTCAR`、`vasprun.xml` 等允许文件，查看收敛和磁矩解析结果。如果同时提供 `TDOS.dat`，页面绘制 DOS；能带图需要 `BAND.dat` 或 `REFORMATTED_BAND.dat`，以及 `KLABELS`。仅导入 `vasprun.xml` 不会自动产生 DOS/能带图。上述绘图表格可由 Linux/WSL 中的 VASPKIT 或其他兼容流程预先生成。
- **环境**：查看本机 POTCAR 注册表和 Agent 模型配置的就绪状态。

项目、审核记录和导入的结果保存在 `var/web/`，该目录不纳入 Git。Agent 会话及暂停状态目前只保存在服务进程内，重启网页服务后不能从原会话继续。网页尚无首次配置向导、统一集群作业列表或多人账号体系；每人需在自己的电脑上运行。服务只监听 `127.0.0.1`，不要将其作为公网或组内共享服务部署。

导入的 `TDOS.dat`、`BAND.dat`、`REFORMATTED_BAND.dat` 和 `KLABELS` 目前是独立外部表格，网页尚未核对它们与同批 VASP 输出的计算来源、费米能参考或 k 路径是否一致。图像可用于查看这些表格，但不能仅凭同次上传就认定它们属于已解析的 VASP 结果；正式分析前应人工核对来源和能量参考。

## 给组员共享源码

建议使用审查过的 GitHub 私有仓库分发源码。每位组员在自己的电脑上安装依赖，扫描自己的授权 POTCAR 库，配置个人模型密钥和集群身份。源码仓库不应包含 `.env`、`config/agent/llm.json`、个人 `cluster.json` / `identity.json`、POTCAR、运行结果、原始课程资料或 VASPKIT 发行包。VASPKIT 的发行许可证要求未经书面许可不得再分发。
