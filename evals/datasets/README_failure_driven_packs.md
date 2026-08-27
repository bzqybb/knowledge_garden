# 失败驱动题库说明

## 题库清单

| 题库 | 开发集 | 冻结迁移 | 作者可见挑战 | 设计目的 |
|---|---:|---:|---:|---|
| `zhili_structural_debug_v2` | 96 | 32 | 12 | 16 类结构，每类 A–F 对照；验证同结构异表面与新结构迁移 |
| `zhili_foundational_hard_v2` | 30 | 20 | 10 | 数学、物理、化学、生物、计算机科学的 10 类高阶基础推理 |
| `zhili_frontier_guided_reading_v1` | 20 | 10 | 0 | 开发集 10 个概念、20 个异质任务、10 个推理家族；冻结集使用 10 个全新概念与一手来源做跨来源同结构迁移 |

总计 230 题：146 道开发题、62 道冻结迁移题、22 道作者可见挑战题。严格盲测仍需由外部保管方在规则冻结后提供；本仓库不把作者可见挑战冒充严格盲测。

## 生命周期

1. 只在 `development` 上定位重复失败簇。
2. 首错至少跨两题复现后，才修改通用规则或通用路由；不加入题面实体特例。
3. 用原失败题做定向回归，并用未参与修改的开发题检查副作用。
4. 冻结规则版本和代码后，运行 `transfer_validation`。
5. 冻结题一旦用于改规则，必须降级为开发题；不得继续报告为迁移验证。
6. `author_visible_challenge` 只能叫“作者可见新结构快照”。

## 常用命令

```powershell
# 数据与隔离审计
.\.venv\Scripts\python.exe -X utf8 -m evals.structural_debug
.\.venv\Scripts\python.exe -X utf8 -m unittest tests.test_structural_debug -v

# 结构开发、冻结迁移、挑战
.\.venv\Scripts\python.exe -X utf8 -m evals.structural_debug --phase develop --run
.\.venv\Scripts\python.exe -X utf8 -m evals.structural_debug --phase validate --run
.\.venv\Scripts\python.exe -X utf8 -m evals.structural_debug --phase challenge --run

# 基础难题
.\.venv\Scripts\python.exe -X utf8 -m evals.structural_debug --dataset evals\datasets\zhili_foundational_hard_v2\development_30.jsonl --phase develop --run
.\.venv\Scripts\python.exe -X utf8 -m evals.structural_debug --dataset evals\datasets\zhili_foundational_hard_v2\transfer_validation_20.jsonl --phase validate --run
.\.venv\Scripts\python.exe -X utf8 -m evals.structural_debug --dataset evals\datasets\zhili_foundational_hard_v2\author_visible_challenge_10.jsonl --phase challenge --run

# 前沿领读
.\.venv\Scripts\python.exe -X utf8 -m evals.structural_debug --dataset evals\datasets\zhili_frontier_guided_reading_v1\development_20.jsonl --phase develop --run
.\.venv\Scripts\python.exe -X utf8 -m evals.structural_debug --dataset evals\datasets\zhili_frontier_guided_reading_v1\transfer_validation_10.jsonl --phase validate --run
```

## 裁判边界

- 本地 `observable_checks` 只检查回答是否有实质内容、推理连接和对应题型动作，不是语义正确率。
- 参考答案、`common_failures` 和 `scoring_rubric` 在回答完成后才写入裁判报告，不进入被测 Agent 上下文。
- 外部 TokenHub 裁判逐题记录实际模型；Flash 仅在 HTTP 402 时回退到 `deepseek-v4-pro-202606`。
- 开发集分数只能叫调试结果；冻结迁移结果也必须注明是否严格盲测。

## 前沿领读来源策略

前沿题只采用论文、会议原文或研究机构的一手项目报告，题面使用摘要转述而非大段复制。

1. 开发集的 20 道题覆盖主张—证据映射、预测—机制缺口、生成有效性阶梯、标度外推、基准—部署接口、证据成熟度、构念效度、测量误差链、转化证据阶梯和物理约束/OOD 等 10 个家族。
2. 每个家族在两个不同概念上出现，形成“表面不同、结构相同”的对照；细粒度 `reasoning_structure_id` 仍保留任务差异。
3. 冻结集的 10 个 `source_url`、`concept_id` 和 `reading_brief` 均不出现在开发集，`transfer_mode=different_source`。
4. 论文陈述与策展边界分存为 `source_claims`、`reported_evidence`、`curator_inference` 和 `known_unvalidated_links`。
5. 被测 Agent 只接收题目正文；结构 ID、参考、失败标签和 rubric 只在作答后交给裁判。

来源 URL、标题与发布日期保存在每条 JSONL 的 `source_url`、`source_title`、`source_date` 字段中。

## 失败修复层级

| 重复首错 | 优先修复层 | 不应做的替代 |
|---|---|---|
| 不知道、记错、知识过时 | RAG、混合检索、重排与来源审计 | 把实时事实永久写入权重 |
| 封闭题误拒答、结构或方法选错 | 路由、题目正文隔离、推理协议与结构对照示例 | 为每个学科实体追加特例 |
| 数值、符号或代码执行错误 | 计算器、Python、形式验证器及结果回读 | 奖励更长但不可验证的推理文本 |
| 条件遗漏、结论越界 | 独立审题 Agent、原子 rubric、失败判据 | 让答题 Agent 用自己的答案自证 |
| 同一结构首错跨领域稳定复现 | 少量高质量对照 SFT | 堆叠海量同义题 |
| SFT 后仍有可客观验证的策略缺陷 | 基于正确性、条件完整性和工具一致性的 RL/GRPO | 直接奖励思考长度 |
