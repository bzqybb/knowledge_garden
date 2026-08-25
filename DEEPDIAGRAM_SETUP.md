# DeepDiagram 本机接入

知识花园采用完整 DeepDiagram 后端 + 自有安全 SVG 渲染器：DeepDiagram 负责选择图解 Agent、生成设计方案和图代码；知识花园只接收经过解析、来源过滤和结构校验的节点/边，不执行任意 HTML 或脚本。

## 当前安装

- 官方源码：`vendor/DeepDiagram/`
- 独立 Python：`vendor/DeepDiagram/backend/.venv/`
- 本地后端：`http://127.0.0.1:8000`
- 本地数据库：`vendor/DeepDiagram/backend/deepdiagram.db`
- 等待时间：默认 45 秒，可用 `DEEPDIAGRAM_TIMEOUT_SECONDS` 调整
- 许可证：上游 DeepDiagram 使用 AGPL-3.0

为降低比赛电脑的部署负担，本项目没有安装 DeepDiagram 自带的 React 前端，也没有运行 Docker/PostgreSQL；知识花园只需要它的 FastAPI 生成服务。SQLite 兼容层只在 SQLite 模式跳过 PostgreSQL 专用迁移，模型、Agent、路由和 SSE 生成链路保持上游实现。

## 启动和识别

双击 `启动知识花园.cmd` 即可。`run.ps1` 会先检查 8000 端口：

- 健康：显示 `DeepDiagram full service connected.`
- 未启动但已安装：自动后台启动后端
- 启动失败：知识花园继续运行，并使用本地确定性图解回退

页面图解卡片右上角可区分真实执行路径：

- `完整 DeepDiagram`：完整服务生成并通过花园审查
- `本地图解回退`：完整服务未交付或产物未通过审查

运行日志位于：

- `data/runtime/deepdiagram.log`
- `data/runtime/deepdiagram-error.log`

## 隐私边界

DeepDiagram 服务只监听 `127.0.0.1`。真正生成图解时，知识花园会把本轮经过审核的内容蓝图和本轮使用的模型配置提交给本机 DeepDiagram；DeepDiagram 再调用所配置的 OpenAI-compatible 模型。它不会在启动时扫描整个 Obsidian 知识库。
