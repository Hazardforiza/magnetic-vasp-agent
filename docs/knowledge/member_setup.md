# 组员本机配置与使用记录

本组共用同一套 `mu01` 集群和计算节点白名单。每人在自己的电脑上安装控制面，填自己的集群用户名和连法。不要复制别人的 `identity.json` 或已合成的 `cluster.json`。

## 全组共用（可进仓库）

- 站点模板：`config/execution/site.team.json`
- 节点：`cu20` `cu21` `cu22` `cu23` `cu38` `cu39` `cu40` `cu41` `cu43` `cu48` `cu49` `cu50`
- 跳板短主机名：`mu01`（禁止在其上跑 VASP / MPI）
- 无 Slurm；单节点 Intel MPI + VASP 5.4.4
- 工位直连：`192.168.164.203:7722`（connection id `office`）
- 本机隧道：`127.0.0.1:17722`（connection id `tunnel`）。隧道由这台电脑自己维护，站点文件不记录 ZeroTier 或密码。

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

