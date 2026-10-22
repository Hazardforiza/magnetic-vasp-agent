# 组员本机配置与使用守则

发布仓库里的站点配置是不可直接连接真实集群的占位模板。管理员先确认站点地址、计算节点白名单、软件版本和绝对路径；每位组员再在自己的电脑上配置身份。不要复制别人的身份文件、合成 profile、密钥或运行目录。

## 共享模板

- `config/execution/site.team.json`：仅含示例值，真实站点值由管理员在受控环境提供。
- `config/execution/cluster.example.json`：完全自定义站点时可用的通用 profile 示例。
- `config/execution/identity.example.json`：每人私有身份文件的起点。

不能把示例网关、节点、模块、端口或路径当作已获准的站点配置。使用前须与管理员核对。

## 每人本机私有

复制 identity 模板：

```powershell
Copy-Item config\execution\identity.example.json config\execution\identity.json
```

只填写自己的集群账号和管理员批准的连接 ID。`execution_enabled` 初始必须保持 `false`。身份、SSH 私钥、模型 API key、授权 POTCAR、生成的 `cluster.json` 和计算产物都留在本机，不提交 Git。

用站点模板合成本机 profile：

```powershell
uv run --frozen magvasp execution compose `
  --site config\execution\site.team.json `
  --identity config\execution\identity.json `
  --output config\execution\cluster.json
```

每位使用者应重新合成本机 profile；不要复制他人的 `cluster.json`。远端根目录必须由管理员批准并归属于当前账号。密码只可在人工 SSH 提示中输入，不能保存到命令行参数、环境变量或日志。API key 只从操作系统环境变量读取。

## 使用与知识记录

生成输入包不等于启动计算。由具名真人检查结构、赝势标识、INCAR、KPOINTS、依赖关系和 manifest 哈希；远程执行还需独立资源授权、最新 probe、MPI smoke、受控传输核验及本次启动批准。

审核记录、日志、路径、输出文件和经验笔记不得包含密码、私钥、API key、POTCAR 正文或他人作业目录。个人使用记录保存在本机 `var/` 中；推广到组内知识前须人工脱敏、审查来源和适用条件，并保留“未验证”状态，直到证据足够。
