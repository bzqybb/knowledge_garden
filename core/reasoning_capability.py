from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ReasoningProtocol:
    key: str
    label: str
    decompose: tuple[str, ...]
    derive: tuple[str, ...]
    verify: tuple[str, ...]
    signals: tuple[tuple[str, int], ...]


def _protocol(
    key: str,
    label: str,
    decompose: tuple[str, ...],
    derive: tuple[str, ...],
    verify: tuple[str, ...],
    *signals: tuple[str, int],
) -> ReasoningProtocol:
    return ReasoningProtocol(key, label, decompose, derive, verify, signals)


# The union deliberately contains both the product plan and the categories that
# actually occur in the user-provided benchmark.  Statistical, economic, and
# physical modelling should not be silently squeezed into an adjacent label.
PROTOCOLS: tuple[ReasoningProtocol, ...] = (
    _protocol(
        "integrated_constraints", "综合约束",
        ("列出所有硬约束与信任假设", "找出两两冲突和不可同时满足之处", "区分理论可行与工程可行"),
        ("用反例或下界解释冲突", "列出可选方案分别牺牲什么", "把需求收敛为可验证指标"),
        ("不能用技术名词假装消除代价", "结论必须说明取舍条件"),
        (r"同时(?:无条件)?满足|不可能三角|互相冲突|权衡|威胁模型", 5),
        (r"端到端加密|同态加密|访问模式|任意字段.*搜索", 4),
    ),
    _protocol(
        "counterfactual", "反事实分析",
        ("定义处理、结果与未发生处理时的反事实", "区分前后变化和因果效应", "列出同期变化、选择偏差与溢出"),
        ("选择对照组或准实验识别策略", "写出估计量", "说明平行趋势等识别假设"),
        ("检查报告率、替代结果和效应异质性", "不能把全部观察变化归因于政策"),
        (r"反事实|双重差分|DID|平行趋势|如果没有", 6),
        (r"安装.*后|实施.*后|政策.*前后|下降了?\d+%", 3),
        (r"据此宣称|全部.*归因|这个结论成立吗", 3),
    ),
    _protocol(
        "source_credibility", "信息可信度",
        ("核对每个数字的定义、口径和截止时间", "区分一手来源、转述与不可复核帖子", "查找原始证据链"),
        ("分别陈述已确认下界、较可信估计与未证实说法", "不机械平均不同口径数字"),
        ("标注截至时间", "给出可复核方法和剩余不确定性"),
        (r"来源|官方公告|媒体报道|社交媒体|可信度|到底有多少|尚未证实", 5),
        (r"不同数字|统计口径|核实|查证", 3),
    ),
    _protocol(
        "decision_analysis", "决策分析",
        ("明确目标、约束和评价指标", "区分已知数据、预测与偏好", "比较错误代价与决策可逆性"),
        ("建立条件分支而非虚构缺失数据", "给出阈值、触发条件和阶段性方案"),
        ("做敏感性分析", "说明什么新信息会改变推荐"),
        (r"方案\s*[AB甲乙]|应该选择|如何选择|推荐哪|可逆|迁移成本", 5),
        (r"最适合(?:哪|什么)|不知道.*选|该选什么|选.*还是|在.*之间.*(?:选择|犹豫)", 5),
        (r"如何比较|建议方式", 5),
        (r"成本|收益|风险|阈值|触发条件", 2),
    ),
    _protocol(
        "code_diagnosis", "代码诊断",
        ("给出最小故障场景", "区分症状、根因与触发条件", "标出共享状态和同步边界"),
        ("列出一种真实交错执行", "解释为何现有保护失效", "提出最小修复和更稳健设计"),
        ("修复后检查死锁、丢失更新和性能退化", "不能只凭报错信息猜原因"),
        (r"并发|竞态|race condition|死锁|线程|协程|锁|共享变量", 6),
        (r"代码.*(?:错误|问题|诊断|偶发)|为什么.*报错", 4),
    ),
    _protocol(
        "algorithm_design", "算法设计",
        ("定义输入、输出、约束和问题规模", "选择数据结构", "识别正确性不变量"),
        ("给出伪代码或清晰步骤", "证明终止性与正确性", "分析时间和空间复杂度"),
        ("检查空输入、重复值和极端规模", "复杂度必须对应已定义的规模"),
        (r"算法|伪代码|复杂度|动态规划|贪心|二分|图搜索|数据结构", 6),
        (r"设计.*(?:步骤|方法)|时间复杂度|空间复杂度", 4),
    ),
    _protocol(
        "mathematical_proof", "数学证明",
        ("准确写出命题、量词、数域与必要条件", "先检查命题是否为真", "选择直接证明、反证、归纳或构造"),
        ("列出关键引理", "逐步推出目标且不循环论证", "充要条件分别证明两个方向"),
        ("检查边界与反例", "区分直觉说明和严格证明"),
        (r"证明|当且仅当|充要条件|收敛|存在性|唯一性|反证|归纳", 6),
        (r"证明或反驳|证明或证伪|构造反例", 6),
        (r"极限|逐点|一致收敛", 4),
        (r"定理|引理|命题", 2),
    ),
    _protocol(
        "probability_reasoning", "概率推理",
        ("定义事件和条件", "写明先验、似然与证据", "检查条件概率方向"),
        ("使用贝叶斯公式或全概率公式", "保留归一化分母", "必要时用频数表复核"),
        ("检查基准率忽视", "结论给出条件概率而非确定断言"),
        (r"贝叶斯|先验|后验|条件概率|似然|全概率", 6),
        (r"患病率|灵敏度|特异度|假阳性|检测呈阳性", 6),
        (r"概率|随机变量|独立事件|期望", 3),
    ),
    _protocol(
        "statistical_analysis", "统计分析",
        ("识别总体、样本、指标与分组", "区分描述统计和推断统计", "检查选择偏差、混杂和聚合口径"),
        ("比较分层结果与总体结果", "报告效应量和不确定性", "避免只看显著性"),
        ("检查辛普森悖论、异常值和多重比较", "统计关联不自动构成因果"),
        (r"辛普森|显著性|置信区间|p值|回归|样本|中位数|方差", 6),
        (r"区间(?:很|较)?窄|置信(?:度)?|概率是否可信", 4),
        (r"统计|平均(?:值)?|相关系数|分组数据", 3),
        (r"录取率|申请人数|分部门|总体结论|聚合(?:后|数据)", 5),
    ),
    _protocol(
        "experimental_design", "实验设计",
        ("明确处理、结果变量和目标因果效应", "识别混杂、选择偏差和测量偏差", "定义实验单位"),
        ("设计随机化、对照、盲法或准实验", "预注册主要指标与分析方法", "做功效与样本量考虑"),
        ("检查依从性、失访、污染和外部效度", "观察结果与因果结论分开写"),
        (r"实验设计|对照组|随机分组|随机化|盲法|A/B测试|干预组", 6),
        (r"数据随机切分|数据泄漏|外部验证|新实验室", 6),
        (r"转化为可研究的问题|可研究的?问题|研究问题", 5),
        (r"设计(?:一个|更)?(?:更可靠的)?实验|如何设计.*实验", 6),
        (r"如何(?:切分|安排|验证|设计)|应验证什么|研究与伦理问题", 5),
        (r"实验|样本量|控制变量|处理组", 3),
    ),
    _protocol(
        "causal_inference", "因果推断",
        ("分别写出观察到的相关和待证明的因果命题", "画出处理、结果、混杂与中介关系", "寻找替代解释"),
        ("说明识别策略", "给出该策略依赖的假设", "区分总效应、直接效应和选择效应"),
        ("检查时间顺序、反向因果和遗漏变量", "不能用相关强度替代因果证据"),
        (r"因果|导致|造成|归因|相关不等于因果|混杂|反向因果", 5),
        (r"随机抽签|随机分配|组间.*差异", 5),
        (r"使用.*后.*(?:上升|下降)|前后对比|全校部署", 5),
        (r"影响|下降|上升|增加|减少", 2),
    ),
    _protocol(
        "economic_analysis", "经济学分析",
        ("明确参与者、激励、约束和市场边界", "区分短期与长期、局部与一般均衡", "识别分配与效率目标"),
        ("分析边际变化、替代效应与反馈", "比较政策前后的剩余、外部性和行为反应"),
        ("检查价格与数量的内生性", "数学模型成立不等于现实参数已满足"),
        (r"供给|需求|价格弹性|机会成本|边际|通胀|市场均衡", 6),
        (r"最低工资|就业减少|劳动力市场", 6),
        (r"补贴|税收|租金|工资|失业|福利", 3),
    ),
    _protocol(
        "policy_analysis", "政策分析",
        ("明确政策目标、对象和执行机制", "区分价值取舍与经验预测", "列出利益相关者"),
        ("比较基准情景与替代方案", "分析直接效应、行为反应和执行成本", "给出监测指标"),
        ("说明模型与参数依赖", "检查公平、可执行性和副作用"),
        (r"政策|监管|规制|公共财政|社会福利|政府.*(?:措施|方案)", 5),
        (r"公平|效率|执行成本|利益相关者", 2),
    ),
    _protocol(
        "physical_modelling", "物理建模",
        ("选定系统边界、坐标系和状态量", "列出所用定律、初始/边界条件与近似", "检查量纲"),
        ("从守恒律或运动方程推导", "保留关键中间式", "解释极限情形的物理意义"),
        ("核对单位、符号和数量级", "数学解存在不等于现实模型必然适用"),
        (r"物理建模|量纲|守恒|牛顿定律|运动方程|边界条件", 6),
        (r"碰撞(?!变量)|外冲量|观测方程", 5),
        (r"从.*(?:方程|定律).*推导|推导.*(?:波动方程|运动方程|公式)", 6),
        (r"推导(?:波速|能量|动量|相位)", 5),
        (r"空气阻力|终端速度|自由落体|竖直下落|阻力.*kv", 6),
        (r"速度|加速度|力|能量|动量|电场|磁场|温度|周期", 3),
    ),
    _protocol(
        "logical_reasoning", "逻辑推理",
        ("形式化每个命题和全局约束", "列出所有互斥情况", "确定真值规则"),
        ("穷举、真值表或反证", "逐项淘汰不满足约束的情况", "证明解的唯一性"),
        ("防止漏分支", "不能用直觉猜测代替穷举"),
        (r"恰有|真话|假话|谁偷|谁说谎|逻辑题|真值表|穷举", 6),
        (r"只有一人|唯一(?:可能|解)|甲说|乙说|丙说", 4),
    ),
    _protocol(
        "argument_analysis", "论证分析",
        ("抽取前提、隐含前提和结论", "检查关键词是否偷换含义", "识别全称量词与规范性跳跃"),
        ("构造反例或替代解释", "判断结论强度是否超过前提", "改写为条件更清楚的论证"),
        ("区分反驳结论与指出论证无效", "必须明确缺失的关键前提"),
        (r"论证|偷换概念|隐含前提|反例|推出|因此.*不应该", 6),
        (r"错误回答|请(?:诊断|纠错|修正)|是否(?:正确|成立)", 6),
        (r"前提[：:]", 5),
        (r"前提[：:]?.*(?:判断|评价).*(?:正误|正确|成立)|是否意味着|能否推出|证明完整吗|还缺少什么", 5),
        (r"只(?:给出|知道|报告|测得|测量|使用|在)|仅凭|据此|有限(?:次|个|样本|路径|观测)", 3),
        (r"已知|只知|观察到|实验测得|论文只报告|一次实验|一次.*计时", 3),
        (r"能得出什么|还需了解什么|应如何解释", 4),
        (r"能否|是否|判断", 2),
        (r"任何|所有|必然|绝对权利", 3),
    ),
    _protocol(
        "concept_distinction", "概念辨析",
        ("分别给出最小定义", "找到容易混淆的共同点", "确定真正区分两者的判据"),
        ("用正例、反例或边界案例比较", "说明二者可能重合但不等价"),
        ("避免循环定义", "不要只列外观差异"),
        (r"概念辨析|有什么区别|有何区别|是否等同|是同一个概念|如何区分", 6),
        (r"稳态.*平衡|信度.*效度|可识别|局部.*全局|热力学.*动力学|重复.*复现", 6),
        (r"复现|独立证据|不同实验室|不同仪器|参数.*(?:估计|唯一)|问卷.*测量|表情识别.*理解度", 5),
        (r"有何差别|有何差异", 5),
        (r"什么是|定义|本质区别|异同", 3),
    ),
)


CATEGORY_ALIASES = {
    protocol.label: protocol.key for protocol in PROTOCOLS
} | {
    "代码诊断（并发/竞态）": "code_diagnosis",
    "数学证明（收敛性/存在性）": "mathematical_proof",
    "概率推理（贝叶斯更新）": "probability_reasoning",
    "信息可信度（来源评估）": "source_credibility",
}


def reasoning_subject(question: str) -> str:
    """Remove the structural-eval wrapper before routing or freshness checks.

    Frozen rules can legitimately mention words such as ``最新`` or ``来源``.
    Treating those instructions as part of the learner's question made closed
    proofs look like current-fact requests and triggered evidence refusals.
    """
    text = str(question or "")
    matches = list(re.finditer(r"(?:^|\n)题目[：:]\s*", text))
    return text[matches[-1].end():].strip() if matches else text


def classify_reasoning_task(
    question: str,
    *,
    category_hint: str = "",
    intent_hint: str = "",
) -> dict[str, Any]:
    """Classify a reasoning demand without calling a model or leaking examples.

    A benchmark may provide ``category_hint`` for validation. Product calls do
    not: they are routed only from the user's wording and the already resolved
    high-level intent.
    """
    text = re.sub(r"\s+", "", reasoning_subject(question))
    hinted_key = CATEGORY_ALIASES.get(str(category_hint).strip())
    scores: dict[str, int] = {}
    matches: dict[str, list[str]] = {}
    for protocol in PROTOCOLS:
        score = 0
        found: list[str] = []
        for pattern, weight in protocol.signals:
            match = re.search(pattern, text, re.I)
            if match:
                score += weight
                found.append(match.group(0)[:40])
        if protocol.key == "decision_analysis" and intent_hint == "design":
            score += 1
        if protocol.key in {"argument_analysis", "causal_inference"} and intent_hint == "evaluate":
            score += 1
        if hinted_key == protocol.key:
            score += 100
            found.append("benchmark-category-hint")
        scores[protocol.key] = score
        matches[protocol.key] = found

    protocol = max(PROTOCOLS, key=lambda item: scores[item.key])
    score = scores[protocol.key]
    runner_up = max((value for key, value in scores.items() if key != protocol.key), default=0)
    activated = bool(hinted_key) or score >= 4
    confidence = 1.0 if hinted_key else min(0.95, 0.42 + score * 0.07 + max(0, score - runner_up) * 0.03)
    if not activated:
        confidence = min(confidence, 0.49)
    return {
        **asdict(protocol),
        "activated": activated,
        "score": score,
        "confidence": round(confidence, 3),
        "matched_signals": matches[protocol.key],
        "task_key": f"reasoning:{protocol.key}" if activated else "general",
    }


def is_self_contained_reasoning(question: str, profile: dict[str, Any]) -> bool:
    """Allow deduction from supplied premises without pretending it is sourced fact."""
    if not profile.get("activated"):
        return False
    text = reasoning_subject(question)
    if re.search(r"最新|截至(?:今天|目前|\d{4})|现实中到底|请查|检索|搜索|给出(?:论文|文献|来源|出处)", text):
        return False
    if re.search(r"根据(?:论文|研究|教材|官方数据)|实际统计|真实数据", text):
        return False
    inherently_closed = {
        "mathematical_proof", "probability_reasoning", "algorithm_design",
        "code_diagnosis", "logical_reasoning", "argument_analysis",
        "decision_analysis", "integrated_constraints", "counterfactual",
        "experimental_design", "physical_modelling", "statistical_analysis",
        "causal_inference", "concept_distinction", "economic_analysis", "policy_analysis",
    }
    if profile.get("key") in inherently_closed:
        return True
    return bool(re.search(r"[:：]|\n\s*[-\d]|假设|现有|已知|某(?:公司|城市|平台|研究)", text))


def reasoning_prompt(profile: dict[str, Any], *, surface: str) -> str:
    if not profile.get("activated"):
        return ""
    decompose = "；".join(str(item) for item in profile.get("decompose", []))
    derive = "；".join(str(item) for item in profile.get("derive", []))
    verify = "；".join(str(item) for item in profile.get("verify", []))
    shared = (
        f"【可迁移推理协议：{profile.get('label')}】先在内部完成任务识别和结构拆解。"
        f"拆解重点：{decompose}。关键推导：{derive}。自检：{verify}。"
        "最终只展示用户复核结论所需的关键步骤、公式、真值表、因果关系或算法，不输出无关的私有草稿。"
        "明确区分题设、额外假设、观察结果、推导结论与现实外推；证据不足时给条件性结论，不伪造缺失数据。"
    )
    if surface == "inspiration":
        return shared + (
            "灵感模式保持自然讨论：把假设、替代解释、反例和可观察后果融入正文，不机械套四个固定标题。"
            "只有问题确有唯一或条件性答案时才用 \\boxed{} 收束；开放想法可以明确写‘当前更像待检验假设’。"
        )
    return shared + (
        "问园丁模式在结尾用 \\boxed{...} 标出最终结论；同时紧邻说明成立条件和仍不确定之处。"
        "对比维度重复、穷举分支或多组约束时使用表格；否则不要为格式而造表。关键公式单独成行。"
    )


def review_reasoning_answer(
    profile: dict[str, Any],
    answer: str,
    *,
    surface: str = "gardener_chat",
) -> dict[str, Any]:
    """Cheap hard checks for observable reasoning invariants, not semantic grading."""
    if not profile.get("activated"):
        return {"applicable": False, "passed": True, "checks": {}, "issues": []}
    text = str(answer or "").strip()
    checks = {
        "substantive": len(re.sub(r"\s+", "", text)) >= 80,
        "has_reasoning_link": bool(re.search(r"因此|所以|由此|若|则|推出|=>|⇒|→|=", text)),
        "states_conditions_or_limits": bool(re.search(r"假设|条件|前提|局限|不确定|仅当|除非|仍需|不能", text)),
        "boxed_conclusion": bool(re.search(r"\\boxed\s*\{", text)),
        "not_pseudo_solution": not bool(re.search(r"看起来合理|显然可得|不难发现", text)) or len(text) >= 220,
    }
    issues: list[str] = []
    if not checks["substantive"]:
        issues.append("推理回答过短，尚未给出可复核的关键步骤")
    if not checks["has_reasoning_link"]:
        issues.append("缺少从前提到结论的可见推导连接")
    if not checks["states_conditions_or_limits"]:
        issues.append("没有明确结论成立条件或剩余不确定性")
    if surface == "gardener_chat" and not checks["boxed_conclusion"]:
        issues.append("最终结论未按推理协议使用 \\boxed{} 收束")
    if not checks["not_pseudo_solution"]:
        issues.append("使用了跳步措辞但没有提供足够论证")
    return {
        "applicable": True,
        "passed": not issues,
        "checks": checks,
        "issues": issues,
        "reasoning_type": profile.get("key"),
        "reasoning_label": profile.get("label"),
    }
