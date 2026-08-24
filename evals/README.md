# 知识花园 Ragas 评测

评测分为两层：

1. `retrieval-only`：不调用模型，测当前本地检索的 Recall/Precision。
2. 完整评测：在运行数据库的临时副本上执行 GardenerGraph，再由 Kimi K3 Judge 计算 Context Precision、Context Recall 与 Faithfulness。

## 安装

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-eval.txt
```

## 配置 Kimi K3 Judge

双击 `配置Kimi评测密钥.cmd`，或执行：

```powershell
.\run_evals.ps1 -SaveKimiKey
```

密钥由 Windows DPAPI 加密到 `data/runtime/kimi-eval-api-key.dpapi`，不会进入数据集、报告或源码。也可以临时设置 `KIMI_API_KEY`。默认地址为 `https://api.moonshot.ai/v1`，默认模型为 `kimi-k3`。

## 运行

```powershell
# 不调用模型的检索基线
.\run_evals.ps1 -RetrievalOnly

# 先跑 2 条完整 Kimi Judge 测试
.\run_evals.ps1 -Limit 2

# 跑完整 seed_v1
.\run_evals.ps1
```

报告写入 `evals/reports/`。完整评测使用临时数据库副本，并默认关闭 Wikipedia、OpenAlex 等在线检索，避免污染真实学习记录并保持实验可复现。

Ragas 0.4.3 当前会无条件导入 LangChain 已移除的 VertexAI 旧路径。评测启动器只为这个未使用的可选集成安装进程内兼容占位，不修改第三方包，也不降级知识花园的 LangChain 1.x 依赖。
