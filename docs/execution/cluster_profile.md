# 集群配置契约

执行代码不绑定某个实验室、账号或节点。运行时仍读取合成后的 `ClusterProfile`。本组部署把它拆成两层：

- **站点模板** [`site.team.json`](../../config/execution/site.team.json)：示例节点白名单、`login01`、示例 VASP/MPI 路径、工位直连与本机隧道两种连法、`/home/{user}/magvasp/runs` 目录模板。发布仓库只提供占位模板；实际部署必须由管理员或用户在本机替换端点、节点和软件路径后再启用。
- **个人身份** [`identity.example.json`](../../config/execution/identity.example.json)：集群用户名、选用 `office` 还是 `tunnel`、是否打开执行。复制为 gitignore 的 `identity.json`，不要提交。

```powershell
uv run --frozen magvasp execution compose `
  --site config\execution\site.team.json `
  --identity config\execution\identity.json `
  --output config\execution\cluster.json
```

换完全不同的集群时，仍可使用通用模板 [`cluster.example.json`](../../config/execution/cluster.example.json) 直接写一份 `ClusterProfile`，不必走本组站点模板。

合成后的 profile 至少声明：

- SSH 连接端点、预期跳板主机名、登录用户名和外层/内层端口；
- 允许的计算节点白名单；
- VASP 模块、Intel MPI `mpirun` 的绝对路径和三个 VASP 可执行文件路径；
- 已获本人确认的专用 `remote_root`（必须含自己的用户名路径分量）；
- `execution_enabled` 在未完成组内放行前保持 `false`。

所有主机名、用户名、模块名、节点名和路径都经过 shell-safe 校验。配置文件不能把任意 shell 片段、代理转发或密码注入执行器；密码只允许由人工终端交互输入，永不写入配置或日志。

节点“允许使用”仍必须由集群管理员/组内规则确认。探针的负载、进程列表和 CPU 拓扑只是瞬时观察，不能替代资源分配批准。

## 本机隧道与工位直连（示例）

模板预置两种 connection，均为不可直接连接真实集群的示例值：

- `office`：`gateway.example.org:22`，示例工位网络直连网关。
- `tunnel`：`127.0.0.1:2222`，示例本机 SSH 隧道入口。profile 不记录 overlay 地址或密码。

即使连接端点是本机回环地址，登录后仍必须验证远端短主机名为站点里的 `gateway_hostname`（模板为 `login01`）和当前 identity 中的用户。实际部署要先把模板值替换成管理员确认的值；跳板主机永远不能作为计算节点或 VASP/MPI 提交目标。

需要非默认本机端口时，可在 identity 里覆盖 `port`，不要改全组站点文件。

## 首次登录与后续自动登录

首次验证使用交互模式，由用户在自己的终端输入集群密码：

```powershell
uv run --frozen magvasp execution login --profile config\execution\cluster.json --connect --interactive
```

成功只证明隧道或工位网、账号和跳板身份匹配；密码不会被 Agent 读取或保存。后续无人值守 probe/传输需要公钥认证：用户在实际集群账号和允许的计算节点上按管理员流程安装公钥，或在本机 SSH agent 中加载私钥。Agent 只使用操作系统已经提供的密钥/agent，不接收密码参数，也不启用 agent forwarding 到 `login01`。
