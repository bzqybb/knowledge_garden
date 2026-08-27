# 知识花园 Ragas 评测

## 通用推理 Benchmark（15 类种子集）

用户提供的通用推理题已保存为 `evals/datasets/general_reasoning_15_v1.jsonl`。每题保留题目、参考答案、假设、关键推导、常见错误和带权 rubric；线上 Agent 运行时只接收 `question`，参考答案与 rubric 只在回答完成后进入报告，避免把评测答案泄漏给被测系统。

先运行零模型成本的数据与路由校验：

```powershell
.\run_general_reasoning_benchmark.ps1
```

执行真实 Agent 回答回归（默认禁止联网检索，仍会调用项目已配置的回答模型）：

```powershell
.\run_general_reasoning_benchmark.ps1 -Run
```

可用 `-Limit 3` 或 `-Ids case_001,case_010` 做小样本复测。只有明确需要检索当前事实时才添加 `-AllowNetwork`。每次运行都会在 `evals/reports/` 生成 JSON、JSONL 和 Markdown 三种日志，记录类型路由、耗时、可观察推理检查、实际回答及供人工/独立裁判使用的 rubric。

本地检查只验证“是否有关键步骤、条件/局限和结论收束”等可观察结构，`semantic_score` 固定为空，不能冒充答案正确率。数学正确性、因果识别有效性、算法正确性等仍需隔离的独立裁判或人工复核。

这 15 题参与了能力协议设计，因此属于开发/回归集，不是严格的未知迁移集。要证明泛化，应另建从未用于调优的 `general_reasoning_holdout_v1.jsonl`，保持同一 schema，并在冻结能力实现后一次性运行；不能根据 holdout 单题反复加关键词。

评测分为三层：

1. `retrieval-only`：不调用模型，测当前本地检索的 Recall/Precision。
2. 完整评测：每道题在独立的临时数据库副本上执行 GardenerGraph，由 GLM 生成回答，再由独立裁判计算 Context Precision、Context Recall、Faithfulness、Answer Correctness，以及本地关键知识点覆盖率。
3. 联网研究评测：单独验证百科定位、OpenAlex/Crossref 学术来源与可核验引用。此层默认关闭，必须先明确授权把选定问题发送到对应公开服务。

## 扫描版教材与 54 题书院题库

图片型 PDF 无法通过普通文本提取进入 RAG。知识花园会检测缺失文字层的教材，并使用 Windows 自带中文 OCR 在本机后台识别，自动保存逐页进度；教材不会上传，也不需要额外 OCR API。

```powershell
# 独立运行扫描教材识别；中断后再次执行会从最近的保存页继续
.\.venv\Scripts\python.exe -X utf8 -m evals.ocr_textbooks --books "普通化学,高等代数,复变函数" --rebuild-semantic-index

# 生成 54 道书院题当前的教材覆盖与联网需求快照
.\.venv\Scripts\python.exe -X utf8 -m evals.zhili_three_layer

# 使用上一步输出的 JSONL 跑第一层本地检索
.\.venv\Scripts\python.exe -X utf8 -m evals.run_eval --dataset evals\reports\<覆盖快照>.jsonl --retrieval-only
```

OCR 正在后台进行时，覆盖快照是当前时刻的结果；等新增页持续入库后重新生成，可直接比较新教材对化学、代数与复变函数问题的迁移收益。

## 安装

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-eval.txt
```

## 配置独立 Judge

双击 `配置Kimi评测密钥.cmd`，或执行：

```powershell
.\run_evals.ps1 -SaveJudgeKey
```

密钥由 Windows DPAPI 加密到兼容旧版本的 `data/runtime/kimi-eval-api-key.dpapi`，不会进入数据集、报告或源码。也可以临时设置 `JUDGE_API_KEY`。

当前默认使用腾讯云 TokenHub 的 DeepSeek-V4-Flash 原厂直供（关闭思考模式）：

```powershell
# 只检查密钥、端点和模型是否可用
.\run_tencent_tokenhub_evals.ps1 -Model deepseek-v4-flash-0731 -CheckJudge

# 双击同名 cmd 也可以
.\运行腾讯云TokenHub-DeepSeek-V4-Flash评测.cmd
```

## 运行

```powershell
# 不调用模型的检索基线
.\run_evals.ps1 -RetrievalOnly

# 致理书院基础学科复杂题：数学、力学、电路、计算科学与无教材拒答
.\run_evals.ps1 -Dataset evals\datasets\zhili_foundations_challenge_v1.jsonl -RetrievalOnly

# 先生成 2 条答案，不调用 Judge
.\run_evals.ps1 -Dataset evals\datasets\retrieval_pilot_reviewed_v1.jsonl -Limit 2 -SkipJudge

# 对确认过的答案报告单独评分，避免重复生成和浪费额度
.\run_tencent_tokenhub_evals.ps1 -Model deepseek-v4-flash-0731 -InputReport evals\reports\<报告名>.json -Limit 8
```

## 边界能力、表达自然度与响应耗时

致理书院边界题库包含 22 道校园体验、日常科学、身体不适、哲学思辨和价值判断问题。专项脚本额外加入一道教材可追溯的数学问题，以及两轮偏导数→梯度下降记忆对照，共 25 项。

```powershell
# GLM 生成回答，腾讯云 DeepSeek-V4-Flash 独立审核每一道题
.\.venv\Scripts\python.exe -X utf8 -m evals.boundary_eval --workers 3 --judge-workers 3

# 先试跑两道题，不附加数学和记忆对照
.\.venv\Scripts\python.exe -X utf8 -m evals.boundary_eval --limit 2 --skip-controls

# 对已有回答报告重新调用独立裁判评分，不重复生成答案
.\.venv\Scripts\python.exe -X utf8 -m evals.boundary_eval --input-report evals\reports\<报告名>.json
```

独立裁判对回答有效性、表达自然度、科学严谨、诚实与不确定性、边界安全、真实引用、共情和多轮记忆分别给出 1～5 分，并标注幻觉、失败原因与修改建议。报告同时记录每道题的完整问题、实际回答、引用教材页、各工作流节点耗时以及总体平均/P90 响应时间。评测使用临时数据库副本，不污染真实学习记录；默认关闭外部公开搜索，只调用已经配置的项目回答模型和经用户授权的腾讯云裁判模型。

## 50 道证明、推导、计算与错误前提专项

`evals/datasets/zhili_deep_reasoning_50_v1.jsonl` 完整保留数学、物理、化学、生物、信息与交叉学科的 50 道深度题。每道题都包含参考推理、教材关键词和前提状态。专项另外标记了 9 个容易被模型顺从接受的陷阱，包括恒温恒容误用 Gibbs 自由能、泊松括号遗漏显含时间项、最小多项式未说明分裂域，以及把反向传播与求导错误对立。

```powershell
# 只检查教材覆盖和题目前提，不调用外部模型
.\.venv\Scripts\python.exe -X utf8 -m evals.deep_reasoning_eval --coverage-only

# 项目模型回答全部 50 题并执行本地检查；不会把完整题库发送给独立裁判
.\.venv\Scripts\python.exe -X utf8 -m evals.deep_reasoning_eval --skip-judge --workers 3

# 只把用户授权的少量代表题交给腾讯云独立裁判评分
.\.venv\Scripts\python.exe -X utf8 -m evals.deep_reasoning_eval --ids P-06,P-22,P-28,P-42,P-44,P-45

# 根据裁判建议修改后，只重测失败题并保留其他已有结果
.\.venv\Scripts\python.exe -X utf8 -m evals.deep_reasoning_eval --ids P-28,P-42 --merge-report evals\reports\<之前的报告>.json
```

独立裁判分别检查结论正确性、证明完整性、推导有效性、计算准确性、错误前提纠正、证据忠实度、引用真实性、推理清晰度与不确定性；报告会明确区分“本地教材确实没有覆盖，因此诚实拒答”与“有教材仍然出现幻觉或错误证明”。JSON 的 `actionable_judge_findings` 字段汇总需要优先处理的问题，Markdown 展示全部题目、完整回答、教材页码和具体改进建议。

若用户明确授权补评完整 50 题，可复用已有回答并覆盖最新修复结果，不重复调用项目模型：

```powershell
$env:JUDGE_MODEL='deepseek-v4-flash-0731'
.\.venv\Scripts\python.exe -X utf8 -m evals.deep_reasoning_eval `
  --input-report evals\reports\<50题本地报告>.json `
  --overlay-report evals\reports\<最新修复报告>.json `
  --reset-judge --judge-workers 3
```

报告写入 `evals/reports/`，Markdown 可直接查看每道问题、答案、上下文和分数。完整评测为每道题创建独立临时数据库副本，并默认关闭 Wikipedia、OpenAlex 等在线检索，避免跨题记忆污染、修改真实学习记录或引入不可复现的在线结果。

基础学科挑战集只使用当前已经入库的高等微积分、英文力学、英文电路和本地计算科学概念作为正例，并把尚未导入的化学、生物教材单独列为应拒答的知识边界题。报告会额外按学科与难度汇总 Hit@5、Hit@10、MRR 和教材缺失时的拒答表现。

基础学科的复杂题会自动使用 BGE/FAISS 混合检索和本地精排；真正的短定义题保留快速词法路径。添加或修改教材后，先运行 `python -m evals.build_semantic_index` 增量同步索引。

Ragas 0.4.3 当前会无条件导入 LangChain 已移除的 VertexAI 旧路径。评测启动器只为这个未使用的可选集成安装进程内兼容占位，不修改第三方包，也不降级知识花园的 LangChain 1.x 依赖。
