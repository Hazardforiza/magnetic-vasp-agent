# 组员本机配置与使用记录（示例模板）

发布仓库只包含一套不可直接连接真实集群的示例站点模板。实际部署由管理员提供经过批准的站点值，或由每位用户在自己的电脑上替换模板中的端点、节点和软件路径；每人在自己的电脑上填自己的集群用户名和连法。不要复制别人的 `identity.json` 或已合成的 `cluster.json`。

## 全组共用（可进仓库）

- 站点模板：`config/execution/site.team.json`（仅示例）
- 示例节点：`node01` `node02`
- 示例跳板短主机名：`login01`（禁止在其上跑 VASP / MPI）
- 示例软件：单节点 Intel MPI + VASP 6.4.2；实际版本和绝对路径必须由管理员确认
- 示例工位直连：`gateway.example.org:22`（connection id `office`）
- 示例本机隧道：`127.0.0.1:2222`（connection id `tunnel`）。实际隧道由这台电脑自己维护，站点文件不记录 overlay 地址或密码。

发布模板中的 `gateway.example.org`、`login01`、`node01`、`node02`、软件路径和端口都不是本组真实连接信息。替换前不得启用远程执行。

## 每人本机私有（不进知识库正文、不进 Git）

复制 `config/execution/identity.example.json` 为 `config/execution/identity.json`，只改自己的字段：

- `user`：集群账号
- `connection`：`office` 或 `tunnel`
- `execution_enabled`：未完成组内放行前保持 `false`
- 远端目录默认是 `/home/<user>/magvasp/runs`，不要写成别人的家目录

合成本机运行 profile：

```powershell
uv run --frozen magvasp execution compose `
  --site config\execution\site.team.json `
  --identity config\execution\identity.json `
  --output config\execution\cluster.json `
  --overwrite
```

密码只在这台电脑的 SSH 提示里输入。大模型密钥只放环境变量。POTCAR 只扫描本机授权库。

## 使用知识怎么记

人工审核输入包时，审核人必须是集群用户名这类具名真人。系统会：

1. 追加到本机 `var/reviews/input-reviews.jsonl`
2. 按审核人写入 `var/reviews/usage/<user>.md`（仅本机，`local_only`）

可写入：材料、任务、磁性模式、接受/退回、脱敏 lesson。
不可写入：密码、私钥、API key、POTCAR 正文、他人作业目录。

全组要推广的经验，经人工审查后再写进组策略或已验证案例；不要把个人 usage 文件直接提交到仓库。
