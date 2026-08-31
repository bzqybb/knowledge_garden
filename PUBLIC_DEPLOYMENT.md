# 致知花园公开部署

## 0.2 公测桌面架构

- 测试用户只登录致知花园公测账号，不填写模型 API Key。
- GLM Key 仅由公开服务器进程读取；桌面端只保存经 Windows DPAPI 加密的账号会话令牌。
- 桌面本地的微信、B站、Obsidian 与知识库按账号隔离；只有用户主动用于回答的提问和材料片段经过认证模型代理。
- 模型代理强制使用服务器指定模型，并按账号限制为最多 2 个并发、10 分钟 120 次请求。
- 用户只有明确开启“贡献脱敏样例”后，问题、回答与反馈才进入候选评测池；固定保留集不得用于提示词优化。

当前 `trycloudflare.com` 地址只适合短期邀请测试。持续公测前应改用命名 Cloudflare Tunnel 或其他固定 HTTPS 域名，并把桌面端的 `PUBLIC_BETA_CLOUD_URL` 指向该稳定入口。

## 已具备的公开模式边界

- `GARDEN_AUTH_REQUIRED=true` 时，除静态页面和认证状态外，所有 API 都要求登录。
- 每个账号使用 `GARDEN_DATA_DIR/users/<user-id>/garden.db`，知识、会话、偏好和设置物理隔离。
- 登录 Cookie 使用随机会话令牌、`HttpOnly`、`SameSite=Lax`；公网 HTTPS 部署应设置 `GARDEN_COOKIE_SECURE=true`。
- 微信、Obsidian、本地教材目录和 B 站本地 ASR 在公开模式下拒绝执行，避免把服务器本机当成用户电脑。
- “帮助园丁持续改进”默认关闭。开启后只保存脱敏副本；15% 固定进入 sealed holdout，持续改进脚本不会读取它。

## Render 部署

仓库根目录已经包含 `Dockerfile` 和 `render.yaml`。在 Render 创建 Blueprint 并连接仓库，然后填写：

- `GARDEN_API_KEY`
- `GARDEN_BASE_URL`
- `GARDEN_MODEL`

首次打开站点时数据库中还没有用户，因此可以创建第一个账号。创建后，在 `GARDEN_ALLOW_SIGNUP=false` 的默认配置下注册入口会自动关闭。

SQLite 必须位于持久磁盘 `/var/data`；不要删除 `render.yaml` 中的 disk 配置。免费临时文件系统不适合保存用户数据。

## 本地验证公开模式

```powershell
$env:GARDEN_AUTH_REQUIRED="true"
$env:GARDEN_ALLOW_SIGNUP="false"
$env:GARDEN_COOKIE_SECURE="false"
$env:GARDEN_DATA_DIR="$PWD\data\public-test"
python app.py --host 127.0.0.1 --port 8765
```

本地 HTTP 验证必须保持 `GARDEN_COOKIE_SECURE=false`；正式 HTTPS 必须改为 `true`。

## 持续改进

用户授权后，交互进入中心评测登记库 `GARDEN_DATA_DIR/auth/accounts.db`。运行：

```powershell
.\run_continuous_improvement.cmd --limit 50
```

可配置两个互相独立的裁判：

- `JUDGE_API_KEY`、`JUDGE_BASE_URL`、`JUDGE_MODEL`
- `SECONDARY_JUDGE_API_KEY`、`SECONDARY_JUDGE_BASE_URL`、`SECONDARY_JUDGE_MODEL`

流程只处理 development 样例。确定性检查、两个裁判一致性和用户反馈会生成报告；分歧进入人工复核。脚本不会读取 sealed holdout，也不会修改生产代码、提示词或发布版本。

正式发布候选改进前，仍须运行原始盲测、迁移集和最终验证，再通过小流量灰度发布。

## 桌面伴侣与本地插件

不要要求普通用户在公开服务器上填写本机路径或本机 Token。推荐流程是：

1. 用户先在网站创建账号。
2. 网站“设置”页只提供一个 Windows App 安装包；TraceMemo、Node.js 与 Bilibili MCP 已随包内置，不让用户另行下载插件。
3. 桌面 App 自动启动内置本地组件，首次向导只要求用户选择 Obsidian Vault，并亲自完成微信只读 Token 或 B站扫码授权。
4. 本机资料默认不上传；用户主动选择的片段才进入当前问答。
5. 桌面伴侣通过 Tauri 签名更新包升级，失败时保留当前可运行版本。

脚手架位于 `desktop/`。发行 CI 会验证内置组件、许可证文件和 TraceMemo v2.2.3 官方安装器哈希，主 NSIS 安装器会自动安装该本地组件；缺少任何一项便拒绝产包。首次发布前还必须替换 updater 公钥和 GitHub Releases 地址，并配置 CI 私钥；详见 `desktop/README.md`。
