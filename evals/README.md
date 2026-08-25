# 知识花园 Ragas 评测

评测分为两层：

1. `retrieval-only`：不调用模型，测当前本地检索的 Recall/Precision。
2. 完整评测：每道题在独立的临时数据库副本上执行 GardenerGraph，再由 Kimi K2.6 Judge 计算 Context Precision、Context Recall 与 Faithfulness。

## 安装

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-eval.txt
```

## 配置 Kimi Judge

双击 `配置Kimi评测密钥.cmd`，或执行：

```powershell
.\run_evals.ps1 -SaveKimiKey
```

密钥由 Windows DPAPI 加密到 `data/runtime/kimi-eval-api-key.dpapi`，不会进入数据集、报告或源码。也可以临时设置 `KIMI_API_KEY`。

当前推荐使用腾讯云 TokenHub 的 Kimi K2.6（关闭思考模式）：

```powershell
# 只检查密钥、端点和模型是否可用
.\run_tencent_tokenhub_evals.ps1 -Model kimi-k2.6 -CheckJudge

# 双击同名 cmd 也可以
.\运行腾讯云TokenHub-K2.6评测.cmd
```

## 运行

```powershell
# 不调用模型的检索基线
.\run_evals.ps1 -RetrievalOnly

# 先生成 2 条答案，不调用 Judge
.\run_evals.ps1 -Dataset evals\datasets\retrieval_pilot_reviewed_v1.jsonl -Limit 2 -SkipJudge

# 对确认过的答案报告单独评分，避免重复生成和浪费额度
.\run_tencent_tokenhub_evals.ps1 -Model kimi-k2.6 -InputReport evals\reports\<报告名>.json -Limit 8
```

报告写入 `evals/reports/`，Markdown 可直接查看每道问题、答案、上下文和分数。完整评测为每道题创建独立临时数据库副本，并默认关闭 Wikipedia、OpenAlex 等在线检索，避免跨题记忆污染、修改真实学习记录或引入不可复现的在线结果。

基础学科的复杂题会自动使用 BGE/FAISS 混合检索和本地精排；真正的短定义题保留快速词法路径。添加或修改教材后，先运行 `python -m evals.build_semantic_index` 增量同步索引。

Ragas 0.4.3 当前会无条件导入 LangChain 已移除的 VertexAI 旧路径。评测启动器只为这个未使用的可选集成安装进程内兼容占位，不修改第三方包，也不降级知识花园的 LangChain 1.x 依赖。
