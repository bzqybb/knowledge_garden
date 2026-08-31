# 可更新桌面伴侣

桌面伴侣采用 Tauri 2：Web UI 留在轻量壳中，现有 Python 服务由 PyInstaller 打包为 sidecar。正式 Windows 安装包同时内置 TraceMemo、Node.js 和固定版本的 Bilibili MCP，用户不需要再下载插件；公开网站继续负责登录、云端问答与授权评测。

首次启动时，伴侣会自动启动内置 TraceMemo。用户仍需亲自完成微信数据只读授权、TraceMemo Token 创建或 B 站扫码登录；这些凭据按用户隔离保存在本机，不能打进安装包。公测用户只需登录知识花园账号：模型 API Key 始终保存在公测服务器，桌面端只用 Windows DPAPI 加密保存用户自己的短期会话令牌。

## 首次发布前必须完成

1. 保持发行范围为比赛、测试和非商业展示；审查记录见 `NONCOMMERCIAL_DISTRIBUTION.md`。CI 已锁定 TraceMemo v2.2.3 官方 Windows 安装器及 GitHub 公布的 SHA-256，摘要不符会主动失败。任何商业化版本必须重新取得书面许可。
2. 在 `src-tauri/tauri.conf.json` 替换 GitHub Releases 地址。
3. 运行 `pnpm run tauri signer generate -- -w <安全位置>` 生成更新签名密钥。
4. 把公钥内容写入 `plugins.updater.pubkey`；私钥只放到 CI Secret `TAURI_SIGNING_PRIVATE_KEY`，绝不能提交。
5. Windows 正式分发还应配置代码签名证书，避免 SmartScreen 把安装包标为未知发布者。

## Windows 本地构建

```powershell
cd desktop
.\prepare_bilibili_runtime.ps1
$env:TRACEMEMO_REDISTRIBUTION_APPROVED = "true"
$env:TRACEMEMO_BUNDLE_PATH = "D:\approved\tracememo-2.2.3-setup.exe"
$env:TRACEMEMO_SHA256 = "3c0c89e463ea4acd5bbf36e2fbfbe72ad49bce6aaa39913b765c5ac596804dff"
.\prepare_bundled_components.ps1
.\verify_release_components.ps1
.\build_sidecar.ps1
# 本机测试包：不生成需要私钥签名的 updater artifact
$env:GARDEN_BETA_CLOUD_URL = "https://你的固定公测域名"
.\build_desktop.cmd
```

公测服务器地址在编译期注入，源码和安装包中都不保存模型 API Key。没有设置
`GARDEN_BETA_CLOUD_URL` 时，桌面端仍可作为纯本地版运行，但不会启用公测账号和服务器代付模型。

`build_desktop.cmd` 会在当前进程内把 pnpm、Cargo、PyInstaller、模型与临时缓存统一指向项目的 D 盘 `data/`，不会改写 Windows 全局 `TEMP`。已经安装依赖时可使用 `powershell -ExecutionPolicy Bypass -File .\build_desktop.ps1 -SkipInstall`。

本地测试包尚未做 Windows 代码签名，因此 SmartScreen 可能显示“未知发布者”。正式 CI 不使用 `tauri.local.conf.json`：Tauri updater 只接受签名更新包，稳定通道使用 GitHub Releases 的 `latest.json`；需要测试通道时应使用另一个签名发布地址，而不是让网页下发可执行代码。
