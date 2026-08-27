from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
HARD_PACK = ROOT / "evals" / "datasets" / "zhili_foundational_hard_v2"
FRONTIER_PACK = ROOT / "evals" / "datasets" / "zhili_frontier_guided_reading_v1"


def group(
    sid: str,
    title: str,
    discipline: str,
    rule: str,
    development: list[tuple[str, str, str, list[str]]],
    validation: list[tuple[str, str]],
) -> dict[str, Any]:
    return {
        "structure_id": sid,
        "title": title,
        "discipline": discipline,
        "rule": rule,
        "development": development,
        "validation": validation,
    }


HARD_GROUPS = [
    group("HF01", "逐点收敛、一致收敛与极限换序", "数学分析",
          "先写出量词次序；交换极限、积分或导数必须核对一致性或支配条件。", [
        ("A", "设 f_n(x)=x^n（0≤x≤1）。证明它逐点收敛，并判断是否一致收敛。", "逐点极限在[0,1)为0、在1为1；取接近1的点可知上确界误差不趋零，故不一致。", ["只算逐点极限", "忽略端点附近的上确界"]),
        ("C", "设 f_n:[0,1]→R 连续，且对每个 x∈[0,1] 都有 f_n(x)→f(x)。能否必然推出 ∫_0^1 f_n(x)dx→∫_0^1 f(x)dx？若不能，给出连续函数反例，并写出一组保证换序成立的充分条件。", "不能；可取支撑在[1/n,2/n]、高度为 n、面积为1的连续三角尖峰，它逐点趋于0但积分恒为1；一致收敛，或满足可积支配函数等支配收敛条件，均可保证积分换序。", ["把逐点收敛当一致收敛", "没有给出反例或可用的充分条件"]),
        ("D", "证明：若每个 f_n 都连续且逐点收敛，则极限函数必连续。", "命题错误；连续函数列的逐点极限可不连续，例如 x^n 在[0,1]上的极限。", ["顺从错误命题"]),
    ], [
        ("若 f_n 在紧集上一致收敛且每个 f_n 连续，极限是否连续？说明证明骨架。", "是；用一致误差和单个连续函数的局部误差作三角不等式。"),
        ("函数项级数可逐项求导需要哪些比逐点收敛更强的条件？", "常用条件是导数级数一致收敛且在一点函数值收敛，再推出原级数一致收敛并可逐项求导。"),
    ]),
    group("HF02", "可对角化、正规与谱定理", "线性代数",
          "先明确数域与内积结构；可对角化、酉对角化和正规性不是同一层结论。", [
        ("A", "证明复内积空间中的正规矩阵可以酉对角化，并指出关键不变子空间。", "用 Schur 酉三角化；正规上三角矩阵必为对角矩阵，或用特征向量正交补的不变性归纳。", ["只说有特征值", "未得到正交基"]),
        ("B", "一个实矩阵在复数域可对角化，是否必能被实正交矩阵对角化？", "不必；实正交对角化要求实对称。旋转矩阵在复域可对角化但不能实正交对角化为实对角形。", ["忽略数域", "把可对角化等同正交对角化"]),
        ("C", "设 A∈C^{n×n}。已知其特征多项式有 n 个根，计代数重数。能否据此断言 A 可对角化？若不能，给出一个 2×2 反例，并写出一个等价判据。", "不能；复矩阵的特征多项式本来就有 n 个根，计重数；Jordan 块 [[λ,1],[0,λ]] 不可对角化；可对角化当且仅当各特征空间维数之和为 n，等价地最小多项式无重根。", ["把代数重数当几何重数", "遗漏数域或反例"]),
    ], [
        ("厄米矩阵为何一定有实特征值且不同特征值的特征向量正交？", "分别由内积共轭对称与自伴关系推出实谱和正交性。"),
        ("Jordan 块大小与最小多项式重根有什么对应关系？", "每个特征值对应最大 Jordan 块大小等于最小多项式中该因子的指数。"),
    ]),
    group("HF03", "对称性、守恒量与适用条件", "理论力学",
          "识别作用量的连续对称及边界项；守恒量依赖动力学和对称条件，不由表面几何自动给出。", [
        ("A", "设 L(q,q̇,t)∈C²，且 q(t)∈C² 满足 Euler–Lagrange 方程。证明 E_L=Σ_i q̇_i∂L/∂q̇_i−L 满足 dE_L/dt=−∂L/∂t。说明该恒等式是否需要速度 Hessian 非退化，以及非退化条件在何处才需要。", "对 E_L 求导并用 Euler–Lagrange 方程消去 d/dt(∂L/∂q̇_i)−∂L/∂q_i，得到−∂L/∂t；若 L 不显含时间则 E_L 守恒；恒等式本身不要求速度 Hessian 非退化，非退化用于正规 Legendre 变换和等价 Hamilton 描述；有速度依赖势时 E_L 不能无条件写成通常的 T+V。", ["只背结论", "误称恒等式必须要求 Hessian 非退化", "把能量函数无条件叫机械能"]),
        ("B", "质点在一段时间内沿以原点为圆心的圆周运动。仅凭轨迹为圆，能否推出它关于原点的角动量守恒？给出一个轨迹仍为圆但角动量改变的受力机制。", "不能；径向约束维持圆轨道的同时，切向外力可改变角速度，使 L=mr²θ̇ 改变；角动量守恒需要关于原点的总外力矩为零或相应旋转对称性，而不是仅靠轨迹形状。", ["由形状推守恒", "没有给出切向力矩反机制"]),
        ("D", "证明任何周期运动都必有能量守恒。", "命题错误；受周期外驱动和耗散的稳态运动可周期但机械能持续交换。", ["周期性等同时间平移对称"]),
    ], [
        ("拉格朗日量在连续变换下只改变一个全时间导数时，Noether 结论是否仍成立？", "成立但守恒荷含边界项修正，需把总导数贡献计入。"),
        ("含显式时间依赖约束的系统能量函数为何可能不守恒？", "约束或驱动破坏时间平移对称并可向系统输入或抽取功。"),
    ]),
    group("HF04", "规范势、可观测量与拓扑效应", "电磁学与量子",
          "区分规范依赖的表示与规范不变量；局域场为零不排除全局回路相位。", [
        ("A", "说明 Aharonov–Bohm 相位为何由闭合回路的矢势积分决定，并与磁通联系。", "相位差为 q/ħ∮A·dl=qΦ/ħ；Stokes 关系显示被围区域的磁通可影响相位，即使粒子路径处 B=0。", ["声称局域洛伦兹力导致", "漏掉回路拓扑"]),
        ("B", "改变 A→A+∇χ 会改变波函数相位；这是否意味着可观测干涉条纹任意改变？", "不会；波函数同时作相应局域相位变换，闭合回路相位和可观测概率保持规范不变。", ["把规范量当可观测量"]),
        ("C", "只知某区域 B=0，能否断言该区域内任意 A 都能全局规约为零？", "不能；若区域非单连通，平坦联络仍可能有非平凡环量，需要拓扑信息。", ["把局域结论推广为全局"]),
    ], [
        ("标势和矢势不唯一，为何 E、B 仍唯一？", "规范变换中势的变化相互抵消，使 E=−∇φ−∂A/∂t 与 B=∇×A 不变。"),
        ("磁单极若存在，为何可能需要多个规范片区描述矢势？", "球面上的非平凡纤维结构使单个无奇点全局势不可得，片区间以规范变换粘接。"),
    ]),
    group("HF05", "有限系统、热力学极限与相变", "统计物理",
          "先区分有限配分函数的解析性与热力学极限中的非解析；尖峰和双峰只是有限尺寸证据。", [
        ("A", "设系统只有有限个状态，H_s(λ) 在实参数 λ 的某开区间内解析，且 β>0。证明 Z(λ)=Σ_s exp(−βH_s(λ)) 与 F(λ)=−β^{-1}log Z(λ) 在该区间解析，并说明热力学极限中为何可能出现非解析相变。", "有限个解析正项之和仍解析且 Z>0，因此实对数复合后的 F 解析；自由度趋于无穷时，有限系统自由能的极限可能非一致，复参数零点可向实轴聚积，从而产生非解析性；单说 Z 有限不足以保证解析。", ["把配分函数有限误当作解析性的充分条件", "把数值陡变当非解析证明"]),
        ("B", "有限尺寸模拟出现双峰能量分布，能否直接证明一阶相变？", "不能直接证明；它是线索，还需随系统尺寸的势垒、潜热和标度行为。", ["单一尺寸下结论"]),
        ("C", "已知热容峰随尺寸增高，能否判定临界指数？", "信息不足；需多个尺寸、背景项、有限尺寸标度窗和误差分析。", ["用少量峰值硬拟合"]),
    ], [
        ("二维 Ising 有限晶格磁化曲线平滑，是否反驳热力学极限的临界点？", "不反驳；有限尺寸会圆滑化奇点，需做有限尺寸标度。"),
        ("关联长度接近系统尺度时为何边界条件影响增强？", "系统无法容纳更大尺度涨落，边界和有限尺寸截断临界关联。"),
    ]),
    group("HF06", "稳态、平衡与详细平衡", "物理化学",
          "用通量与熵产生区分稳态和平衡；详细平衡是逐对反应通量平衡，比总浓度不变更强。", [
        ("A", "连续流反应器浓度不随时间变化，说明它达到热力学平衡了吗？", "不说明；进出流和反应可在非零通量下相互抵消形成非平衡稳态。", ["稳态等同平衡"]),
        ("B", "某网络各物种净生成率为零，是否意味着每对正逆反应速率相等？", "不一定；循环通量可使节点净变化为零但不满足详细平衡。", ["总平衡等同逐边平衡"]),
        ("D", "证明催化剂会改变反应的平衡常数。", "命题错误；催化剂改变正逆路径速率和到达平衡的时间，不改变给定温度下平衡常数。", ["混淆动力学与热力学"]),
    ], [
        ("封闭体系达到平衡时为何宏观净通量为零而微观跃迁仍可发生？", "正逆微观事件继续发生但统计通量相互抵消。"),
        ("有持续化学势差维持的网络能否处于平衡？", "通常不能；持续亲和力驱动通量并产生熵，只能是非平衡稳态。"),
    ]),
    group("HF07", "对称性选择定则与微扰破缺", "量子化学",
          "选择定则来自态与算符的对称表示；微扰破缺对称后，原禁戒跃迁可弱允许。", [
        ("A", "用宇称说明中心对称体系中电偶极跃迁为何要求初末态宇称相反。", "矩阵元⟨f|r|i⟩中 r 为奇宇称；被积函数整体为偶才可能非零，所以初末态宇称相反。", ["只背 Δl 规则", "未分析算符对称性"]),
        ("B", "一个在理想对称下禁戒的谱线在实验中很弱但非零，是否推翻选择定则？", "不一定；振动耦合、外场、缺陷或自旋轨道耦合会破缺/混合对称，使跃迁弱允许。", ["把近似规则当绝对禁令"]),
        ("C", "只给分子点群，能否判断某具体跃迁强度？", "不能完整判断；还需初末态不可约表示、跃迁偶极算符分量及态混合、占据等信息。", ["点群标签直接代替矩阵元"]),
    ], [
        ("为什么拉曼活性与红外活性的算符对称性不同？", "红外看偶极矩的一阶变化，拉曼看极化率张量变化，对应不同表示。"),
        ("外电场使禁戒线增强时，最直接的机制检验是什么？", "检验强度随场强的标度与偏振选择，并与场致态混合微扰模型比较。"),
    ]),
    group("HF08", "外显率、上位性与基因因果", "遗传学",
          "把变异—分子功能—表型的因果链分层；关联、共分离和功能扰动提供不同证据。", [
        ("A", "某致病变异的外显率为70%，这是否意味着携带者有30%的细胞没有该变异？", "不是；外显率是携带者群体出现表型的比例，与单个携带者细胞中变异比例不同。", ["混淆群体概率与细胞嵌合"]),
        ("B", "两个基因单独敲除表型轻，双敲除严重，能否直接称为同一路径？", "只能说明非加性遗传互作；同一路径、平行冗余或补偿网络需更多机制实验区分。", ["上位性唯一解释"]),
        ("C", "一个变异在病例中富集且体外影响蛋白活性，能否判定临床致病？", "仍需人群频率、共分离、效应大小、疾病机制、偏倚和独立证据整合。", ["单一功能实验定论"]),
    ], [
        ("同一致病变异在不同家系中的表型差异很大。请从遗传背景、环境、随机或时间过程、检测与入组偏差中任选三个相互不同的层级提出解释，并为每个解释写出一种可区分它的新增数据。", "修饰基因或多基因背景可用全基因组、家系共分离或多基因风险数据区分；环境暴露可用可比较的暴露和生活史数据区分；随机发育、年龄依赖或体细胞过程可用纵向、组织或单细胞数据区分；检测、分型或入组差异可用统一检测流程和病例选择审计区分。"),
        ("孟德尔随机化中的遗传工具变量为何还需排除水平多效性？", "若变异经暴露之外路径影响结局，就违反排除限制并破坏因果解释。"),
    ]),
    group("HF09", "系统发育同源、趋同与水平转移", "进化生物学",
          "树上的相似性需在替代模型、性状演化与基因史/物种史差异下解释。", [
        ("A", "两个远缘物种都有流线体型，能否仅凭形态相似判定最近共同祖先也有该性状？", "不能；相似环境可导致趋同，应结合多个独立性状、化石和分子系统树作祖先状态重建。", ["相似即同源"]),
        ("B", "某基因树与公认物种树冲突，是否必有一个分析错误？", "不必；不完全谱系分选、基因复制丢失、杂交或水平转移可造成真实冲突。", ["基因树等同物种树"]),
        ("C", "只比较一个高度保守基因，能否可靠重建近期物种分化？", "未必；变异信息可能不足，还需多位点、合适模型和取样。", ["一个标记包打天下"]),
    ], [
        ("长枝吸引为何可能把快速进化的非近缘类群聚在一起？", "模型不足时独立同位替代被误当共同衍征，需更好模型和取样。"),
        ("发现细菌基因 GC 含量异常能否证明水平转移？", "只是线索；需系统树冲突、邻域、组成与供体候选等多证据。"),
    ]),
    group("HF10", "可判定性、复杂度与验证", "计算机科学",
          "区分‘存在算法’、‘多项式时间可算’与‘给定答案可高效验证’；归约方向决定结论。", [
        ("A", "解释停机问题不可判定的对角化证明核心，而不是只报结论。", "假设存在判定器 H，构造程序 D 对输入自身做与 H 预测相反的行为；运行 D(D) 得矛盾。", ["循环论证", "未构造自指矛盾"]),
        ("B", "设 L 是判定问题。对每个 yes 实例 x，存在长度至多为 poly(|x|) 的证书 y，且确定性验证器可在 poly(|x|) 时间内判断 R(x,y)。这能推出 L 可由确定性多项式时间算法求解吗？能确定推出什么？", "只能推出 L∈NP；一般不能推出 L∈P，对所有 NP 问题能否由确定性多项式时间求解正是 P 与 NP 问题；验证给定证书不等于高效找到证书。", ["验证等同搜索", "遗漏多项式证书长度或判定问题前提"]),
        ("C", "已知 A 可多项式归约到 B 且 A 很难，能否判断 B 的难度？", "若归约方向为 A≤pB，则 B 至少与 A 一样难；还需明确‘很难’的形式定义。", ["把归约方向说反"]),
    ], [
        ("要证明新问题 X 是 NP 完全，归约应从已知 NP 完全问题到 X 还是反向？", "从已知 NP 完全问题归约到 X，并先证明 X 属于 NP。"),
        ("设一个随机化二元判定算法对每个输入都以至少2/3的概率给出正确答案，不同运行使用独立随机比特。独立运行 k 次并取多数票时，怎样选择 k 才能把错误概率降到至多 δ？", "由 Hoeffding 或 Chernoff 界，多数票错误概率至多 exp(−Ω(k))，故取 k=O(log(1/δ))；必须使用独立运行，且输出须是可多数聚合的二元判定结果。"),
    ]),
]


HARD_CHALLENGE = [
    ("HC01", "局部展开与全局结论", "构造一个 C∞(R) 函数，使它在0点的 Taylor 级数具有无穷收敛半径，但在任何 x≠0 处都不等于原函数，并验证关键步骤。", "取 f(0)=0、f(x)=exp(−1/x²)(x≠0)；它在0点各阶导数均为0，Taylor级数恒为0且处处收敛；x≠0时 f(x)>0；这说明 C∞ 与实解析不同。", ["只报平坦函数而未验证各阶导数", "误称光滑必解析"], "必须同时验证 Taylor 级数与原函数在非零点的差异。"),
    ("HC02", "拓扑不变量与连续变形", "在穿孔平面 R²\\{0} 中构造两条长度都为2π的简单闭曲线，使它们不同伦，并给出判定理由。", "取以原点为圆心的单位圆和以(3,0)为圆心的单位圆；长度均为2π；绕原点的绕数分别为1和0；绕数在避开原点的同伦下不变，故不同伦。", ["只比较长度", "没有给出同伦不变量"], "构造的曲线必须都避开原点，并用绕数或等价不变量判定。"),
    ("HC03", "遍历性与时间平均", "设有限状态马尔可夫链不可约、非周期，平稳分布为 π。对任意有界函数 g，能否推出几乎处处有 T^{-1}Σ_{t=0}^{T-1}g(X_t)→Σ_xπ(x)g(x)？说明所用条件。", "可以；由有限状态遍历定理，时间平均几乎处处收敛到平稳分布下期望；不可约保证唯一平稳分布，非周期保证分布收敛；不能把运行很久单独当遍历性证明。", ["在条件充分时仍回答信息不足", "只说时间足够长"], "结论限于给定有限、不可约条件；应区分时间平均定理与分布收敛。"),
    ("HC04", "手性、旋光与命名", "错误回答：‘R 构型中的 R 就是 right，所以 R 构型一定右旋。’请定位第一处实质错误，并给出正确判定方法。", "首错是把 CIP 构型命名与旋光方向混为一谈；R/S 由取代基优先级和空间排列定义；正负旋光由实验测量或可靠量化计算确定；二者无普遍对应。", ["把 R/S 与正负旋直接对应", "没有指出可操作的旋光判定方法"], "必须区分构型命名规则与实验可观测的旋光符号。"),
    ("HC05", "适应与瞬时响应", "为区分‘刺激已经消失’与‘持续刺激下负反馈适应’，设计一个最小时间序列实验：写出至少两个同时测量的变量和相反预测。", "持续记录外界输入与通路输出，并测一个反馈节点或受体状态；刺激消失时输入读出下降，适应时输入保持而输出回落并伴反馈状态改变；干预反馈可恢复或改变适应轨迹。", ["只测通路输出", "没有给出两种机制的相反预测"], "实验必须同时观测输入与内部状态，且明确可区分的时间序列模式。"),
    ("HC06", "安全归约的方向", "错误证明：‘长期没人找到攻击，所以协议安全；若能把一个已知困难问题归约到该协议，就进一步证明了安全。’请指出两处逻辑错误并写出正确归约方向。", "未发现攻击只是经验性证据，不是安全证明；需先明确攻击者能力、安全游戏与优势界；正确方向是假设存在攻破协议的攻击者，用它构造已知困难问题的求解器；从困难问题归约到协议通常不能证明该安全结论。", ["把未发现攻击当证明", "把归约方向写反"], "安全结论必须相对于明确威胁模型，并写清攻击者到困难问题求解器的构造方向。"),
    ("HC07", "形式验证与规范漏洞", "某程序被证明满足形式规范 S；团队据此断言‘用户需求 R 已满足，因为证明工具是可靠的’。指出结论中缺失的推理环，并说明还需证明什么。", "可靠工具只支持程序满足 S；还缺少 S 正确且完整表达 R 的论证；需做需求追踪、环境假设验证与规范确认，才能把相对规范的正确性连接到真实需求。", ["质疑证明工具却忽略规范—需求缺口", "把相对正确性当需求真实性"], "必须明确区分实现符合规范与规范符合需求。"),
    ("HC08", "逆问题与先验", "给出两个不同真值 x₁≠x₂ 但满足 Ax₁=Ax₂ 的线性逆问题例子，并解释加入最小范数正则化后为何会得到唯一答案却不能证明其细节真实。", "可取有非零零空间的 A，例如 A=[1,0]，x₁=(1,0)、x₂=(1,1)；二者数据相同；最小范数规则选 x₁ 是先验选择产生的唯一性，数据本身没有区分第二分量。", ["没有构造同数据的不同真值", "把正则化唯一性当可识别性"], "必须指出哪些分量由数据识别、哪些来自先验。"),
    ("HC09", "普适性与微观机制", "给出两个微观相互作用不同但可属于同一普适类的系统，并列出判定同一普适类需要匹配的三个宏观结构条件。", "例如三维单轴铁磁体与三维 Ising 格点模型；需匹配空间维数、序参量对称性以及相互作用的有效程或守恒结构；相同临界指数支持共同长程标度，不推出微观相互作用相同。", ["只说临界指数相同", "把普适类等同微观模型"], "例子与条件必须对应同一临界固定点的长程结构。"),
    ("HC10", "量子层析与测量完备性", "对量子比特测得 ⟨σ_x⟩=0.6、⟨σ_y⟩=−0.2、⟨σ_z⟩=0.4。重建密度矩阵，并检查它是否为合法量子态。", "用 ρ=(I+r·σ)/2 得 ρ=[[0.7,0.3+0.1i],[0.3−0.1i,0.3]]；其迹为1且 Hermitian；|r|=sqrt(0.56)<1，故半正定，是合法混态。", ["非对角元的 σ_y 符号错误", "未检验半正定性"], "必须同时满足 Hermitian、迹为1与 Bloch 向量长度不超过1。"),
]

HARD_ASSERTION_SPECS = {
    "HC02": {
        "checker": "circle_winding",
        "schema": {
            "curves": [
                {"center": [0, 0], "radius": 0, "orientation": "ccw"},
                {"center": [0, 0], "radius": 0, "orientation": "ccw"},
            ],
            "claimed_lengths": [0, 0],
            "claimed_winding_numbers": [0, 0],
        },
    },
    "HC10": {
        "checker": "qubit_density_matrix",
        "schema": {
            "rho": [
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ],
        },
    },
}


FRONTIER_SOURCES = [
    {
        "id": "FG01", "concept": "AlphaFold 3 与生物分子相互作用结构预测", "discipline": "计算生物学",
        "source_title": "Accurate structure prediction of biomolecular interactions with AlphaFold 3",
        "source_url": "https://www.nature.com/articles/s41586-024-07487-w", "source_date": "2024-05-08",
        "brief": "论文提出以扩散式生成模块统一预测蛋白质、核酸、小分子配体与离子组成的复合物结构，并在多类基准上报告提升；这些结果是结构预测基准，不等同于亲和力、动力学或细胞内因果机制验证。",
    },
    {
        "id": "FG02", "concept": "AlphaGenome 与长程调控序列建模", "discipline": "基因组学",
        "source_title": "Advancing regulatory variant effect prediction with AlphaGenome",
        "source_url": "https://www.nature.com/articles/s41586-025-10014-0", "source_date": "2026-01-28",
        "brief": "论文用统一模型从最长约一百万碱基的序列预测多种功能基因组信号，并评估变异效应；高预测性能可生成调控假设，但本身不证明分子机制，也不能直接替代临床致病性判定。",
    },
    {
        "id": "FG03", "concept": "Evo 2 与跨生命域基因组基础模型", "discipline": "计算基因组学",
        "source_title": "Genome modelling and design across all domains of life with Evo 2",
        "source_url": "https://doi.org/10.1038/s41586-026-10176-5", "source_date": "2026-03-04",
        "brief": "论文训练跨生命域的长上下文基因组模型，报告序列理解、变异效应与序列生成任务，并开放部分资源；生成的序列具有统计与功能线索，不自动等于安全、可表达或在生物体内有效。",
    },
    {
        "id": "FG04", "concept": "低于阈值的表面码量子纠错", "discipline": "量子信息",
        "source_title": "Quantum error correction below the surface code threshold",
        "source_url": "https://www.nature.com/articles/s41586-024-08449-y", "source_date": "2024-12-09",
        "brief": "实验比较不同码距的表面码存储并观察逻辑错误随码距增加而受抑，构成低于阈值运行证据；仍存在稀有相关错误、解码与大规模资源开销，不能由此宣布容错量子计算已经完成。",
    },
    {
        "id": "FG05", "concept": "GenCast 概率天气预报", "discipline": "地球系统科学",
        "source_title": "Probabilistic weather forecasting with machine learning",
        "source_url": "https://www.nature.com/articles/s41586-024-08252-9", "source_date": "2024-12-04",
        "brief": "GenCast 以条件扩散生成全球十五天集合预报，在论文设定的1320个目标中的大多数优于当时 ECMWF ENS，并显著加速采样；评测依赖再分析初始化和既定指标，业务部署还需实时资料、稳健性和决策效用验证。",
    },
    {
        "id": "FG06", "concept": "AI co-scientist 多智能体科学假设生成", "discipline": "AI for Science",
        "source_title": "Accelerating scientific breakthroughs with an AI co-scientist",
        "source_url": "https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/", "source_date": "2025-02-19",
        "brief": "官方研究报告描述由多个专门智能体生成、批评和排序科学假设及研究方案，并给出若干专家与实验案例；这是辅助提出假设的系统证据，不等同于自动完成发现，且需要防止文献偏差、虚构依据和评价泄漏。",
    },
    {
        "id": "FG07", "concept": "脑类器官中的可塑性与学习样行为", "discipline": "神经科学",
        "source_title": "Human neural organoid microphysiological systems show the building blocks necessary for basic learning and memory",
        "source_url": "https://www.nature.com/articles/s42003-025-08632-5", "source_date": "2025-08-16",
        "brief": "研究测量人神经类器官的受体表达、网络连接、临界性及刺激诱导的短期和长期突触可塑性，把它们作为学习与记忆的基础构件；这些细胞和网络指标不能直接推出意识、主观体验或与完整人脑等价。",
    },
    {
        "id": "FG08", "concept": "亚细胞空间转录组 PHOTON", "discipline": "空间组学",
        "source_title": "Subcellular level spatial transcriptomics with PHOTON",
        "source_url": "https://www.nature.com/articles/s41467-025-59801-3", "source_date": "2025-05-14",
        "brief": "PHOTON 将高分辨成像与高通量测序结合以读取目标细胞或亚细胞区室的转录组，并用实例展示 RNA 定位研究；分辨率、捕获效率、区室选择与图像配准仍会影响生物解释。",
    },
    {
        "id": "FG09", "concept": "Prime editing 精准基因编辑", "discipline": "分子生物学",
        "source_title": "Search-and-replace genome editing without double-strand breaks or donor DNA",
        "source_url": "https://www.nature.com/articles/s41586-019-1711-4", "source_date": "2019-10-21",
        "brief": "论文用 Cas9 nickase、逆转录酶和 pegRNA 在多种人细胞中实现替换、插入和删除，并报告若干位点副产物较少；体外细胞结果不直接保证体内递送、组织特异性、长期安全或临床收益。",
    },
    {
        "id": "FG10", "concept": "Fourier Neural Operator 学习偏微分方程解算子", "discipline": "科学机器学习",
        "source_title": "Fourier Neural Operator for Parametric Partial Differential Equations",
        "source_url": "https://openreview.net/forum?id=c8P9NQVtmnO", "source_date": "2021-05-06",
        "brief": "论文在函数空间层面学习参数到 PDE 解的算子，用傅里叶域参数化获得跨离散分辨率能力，并在若干 PDE 基准报告速度与误差优势；这不保证任意方程、边界条件或分布外参数上的守恒与稳定。",
    },
]


FRONTIER_SOURCE_AUDIT = {
    "FG01": {
        "source_claims": ["统一模型可预测多类生物分子复合物结构。"],
        "reported_evidence": ["论文报告多类结构基准与实验结构对照。"],
        "curator_inference": ["结构预测能力与亲和力、动力学、细胞内因果机制不是同一主张。"],
        "known_unvalidated_links": ["亲和力", "动力学", "细胞内机制"],
        "source_type": "peer_reviewed_article",
    },
    "FG02": {
        "source_claims": ["长上下文序列模型可预测多类功能基因组信号与变异效应。"],
        "reported_evidence": ["论文在预先设定的基因组预测任务上报告性能。"],
        "curator_inference": ["预测效应不自动等于调控机制或临床致病性。"],
        "known_unvalidated_links": ["因果调控机制", "跨人群稳健性", "临床分类"],
        "source_type": "peer_reviewed_article",
    },
    "FG03": {
        "source_claims": ["跨生命域长上下文模型可建模并生成基因组序列。"],
        "reported_evidence": ["论文报告序列理解、变异效应与生成任务结果。"],
        "curator_inference": ["统计像真只是生成有效性的第一层。"],
        "known_unvalidated_links": ["表达", "生物功能", "跨宿主安全"],
        "source_type": "peer_reviewed_article",
    },
    "FG04": {
        "source_claims": ["表面码逻辑错误可在所测码距范围随码距增加而受抑。"],
        "reported_evidence": ["实验比较不同码距的逻辑存储错误。"],
        "curator_inference": ["有限码距低于阈值证据不等于完整容错计算。"],
        "known_unvalidated_links": ["稀有相关错误", "解码扩展", "全系统资源"],
        "source_type": "peer_reviewed_article",
    },
    "FG05": {
        "source_claims": ["扩散集合模型在论文设定的大多数天气目标上优于比较系统。"],
        "reported_evidence": ["论文使用再分析初始化和既定概率预报指标比较。"],
        "curator_inference": ["离线指标优势与实时业务决策效用需分开验证。"],
        "known_unvalidated_links": ["实时资料", "分布漂移", "预警决策效用"],
        "source_type": "peer_reviewed_article",
    },
    "FG06": {
        "source_claims": ["多智能体系统可生成、批评和排序科学假设与方案。"],
        "reported_evidence": ["官方报告给出专家评价和若干实验案例。"],
        "curator_inference": ["官方案例的证据成熟度低于独立盲化复现。"],
        "known_unvalidated_links": ["评价泄漏", "虚构依据", "独立实验增益"],
        "source_type": "official_research_blog",
    },
    "FG07": {
        "source_claims": ["人神经类器官呈现受体、网络与刺激诱导可塑性指标。"],
        "reported_evidence": ["论文测量短期和长期突触与网络变化。"],
        "curator_inference": ["学习样代理指标不能直接推出意识或完整人脑等价。"],
        "known_unvalidated_links": ["机制特异性", "跨实验室复现", "主观体验"],
        "source_type": "peer_reviewed_article",
    },
    "FG08": {
        "source_claims": ["PHOTON 可获得目标细胞或亚细胞区室的转录组。"],
        "reported_evidence": ["论文结合高分辨成像、测序与实例定位分析。"],
        "curator_inference": ["空间定位结果受捕获、分割、配准与区室污染影响。"],
        "known_unvalidated_links": ["绝对捕获效率", "定位因果功能", "跨样本稳健性"],
        "source_type": "peer_reviewed_article",
    },
    "FG09": {
        "source_claims": ["Prime editing 可在多种人细胞位点实现替换、插入和删除。"],
        "reported_evidence": ["论文报告细胞体系中的目标编辑与副产物。"],
        "curator_inference": ["体外编辑到临床收益之间仍有多级证据缺口。"],
        "known_unvalidated_links": ["体内递送", "组织特异性", "长期安全", "临床收益"],
        "source_type": "peer_reviewed_article",
    },
    "FG10": {
        "source_claims": ["FNO 可在函数空间学习若干参数化 PDE 的解算子。"],
        "reported_evidence": ["论文报告若干 PDE 基准上的误差、速度与跨网格表现。"],
        "curator_inference": ["跨网格不等于跨方程、跨边界或 OOD 稳定。"],
        "known_unvalidated_links": ["守恒", "长期稳定", "分布外参数", "任意 PDE"],
        "source_type": "peer_reviewed_conference_paper",
    },
}


FRONTIER_DEV_TASKS = {
    "FG01": [
        {
            "suffix": "A", "reasoning_structure_id": "FR01_CLAIM_EVIDENCE_MAP",
            "prompt": "把材料拆成‘模型产物→直接比较证据→允许主张’三段链；每一箭头写明对象与指标。",
            "reference": "模型产物是复合物结构预测；直接证据是与实验结构的基准比较；只可支持所测复合物类别上的结构预测表现，不能由结构误差直接推出亲和力、动力学或细胞内机制。",
            "failures": ["把结构预测写成亲和力预测", "没有指出直接比较对象"],
            "boundary": "主张必须停留在所测结构任务与复合物类别。",
        },
        {
            "suffix": "E", "reasoning_structure_id": "FR08_MEASUREMENT_PIPELINE_ERROR",
            "prompt": "若高置信度预测在配体结合位点仍可能错，列出两种‘总体分数看不见、局部任务很关键’的误差，并说明应增加什么读出。",
            "reference": "总体骨架正确仍可能有配体姿态、界面侧链或离子配位错误；应增加界面/配体局部误差、构象簇与置信度校准读出，而不能只报单一全局结构分数。",
            "failures": ["只重复总体精度", "把置信度当正确性证明"],
            "boundary": "局部读出只能验证对应结构细节，不能替代功能实验。",
        },
    ],
    "FG02": [
        {
            "suffix": "A", "reasoning_structure_id": "FR02_PREDICTION_MECHANISM_GAP",
            "prompt": "模型准确预测某变异的表达方向。写出从‘预测正确’到‘调控机制成立’之间至少两个仍缺失的因果环。",
            "reference": "仍需在相关细胞类型扰动该变异并测表达或染色质读出，还需定位介导效应的调控元件、因子或接触；预测准确本身不能识别唯一机制。",
            "failures": ["把相关预测当机制证明", "未提出干预证据"],
            "boundary": "机制主张须由干预和中介链支持。",
        },
        {
            "suffix": "E", "reasoning_structure_id": "FR09_TRANSLATIONAL_EVIDENCE_LADDER",
            "prompt": "把‘序列预测→功能验证→临床致病性’写成三级证据阶梯，并指出每级新增的证据类型。",
            "reference": "第一级是预训练任务或独立预测性能；第二级需相关细胞/组织中的功能扰动与复现；第三级还需人群频率、共分离、疾病机制和临床标准整合。",
            "failures": ["用模型分数直接做临床定论", "遗漏人群与共分离证据"],
            "boundary": "模型输出只是临床证据链的一部分。",
        },
    ],
    "FG03": [
        {
            "suffix": "A", "reasoning_structure_id": "FR03_GENERATION_VALIDITY_LADDER",
            "prompt": "为生成序列建立‘统计像真→可表达→有目标功能→可接受安全性’四级验证阶梯，并写出任一级失败对结论的影响。",
            "reference": "序列分布相似只支持统计像真；表达需宿主或无细胞读出；功能需任务特异实验；安全需宿主负担、脱靶或风险评估；任一级失败都阻断更高层主张。",
            "failures": ["生成得像就当有功能", "跳过表达层"],
            "boundary": "不得从序列似然直接跨越到生物体内有效或安全。",
        },
        {
            "suffix": "E", "reasoning_structure_id": "FR07_NEGATIVE_RESULT_UPDATE",
            "prompt": "若生成序列能稳定表达但无目标功能，哪些原主张仍保留，哪些必须撤回？给出更新后的最强可支持结论。",
            "reference": "可保留可生成且可表达的结论；必须撤回目标功能与更高层效用主张；更新结论应限定为该宿主和表达条件下获得稳定产物但未显示目标功能。",
            "failures": ["把一个负结果说成模型完全无用", "仍宣称功能成功"],
            "boundary": "负结果只更新被直接检验的层级及其上游外推。",
        },
    ],
    "FG04": [
        {
            "suffix": "A", "reasoning_structure_id": "FR04_SCALING_EXTRAPOLATION",
            "prompt": "解释‘所测码距下逻辑错误随码距下降’为何是阈值证据，却还不是大规模容错计算证明；写出外推所缺的两项条件。",
            "reference": "有限码距标度支持在所测错误模型和解码器下低于阈值；仍缺稀有相关错误的尾部控制，以及逻辑门、解码时延和全系统资源随规模扩展的证据。",
            "failures": ["把逻辑存储等同通用容错计算", "忽略相关错误"],
            "boundary": "阈值结论须绑定所测码距、错误模型和解码器。",
        },
        {
            "suffix": "E", "reasoning_structure_id": "FR10_RARE_FAILURE_STRESS",
            "prompt": "设计一个专门寻找稀有相关错误的压力测试：写出对照、尾部读出和会推翻简单独立错误模型的结果模式。",
            "reference": "在相同平均物理错误率下比较受控相关噪声与近独立噪声；读取逻辑错误等待时间或高分位尾部及其随码距标度；若尾部显著变重或增码距不再抑制错误，则独立错误外推失败。",
            "failures": ["只看平均错误率", "没有推翻条件"],
            "boundary": "压力测试检验特定相关噪声，不代表枚举所有硬件失效模式。",
        },
    ],
    "FG05": [
        {
            "suffix": "A", "reasoning_structure_id": "FR05_BENCHMARK_DEPLOYMENT_GAP",
            "prompt": "列出从再分析初始化的离线概率基准到实时业务部署之间三处可能改变结论的接口。",
            "reference": "至少包括实时观测与同化延迟、分布漂移和极端事件校准、以及预报指标到具体决策损失的映射；离线多数指标获胜不自动保证业务效用。",
            "failures": ["只说需要更多数据", "把计算快等同决策好"],
            "boundary": "业务结论须在实时输入和任务损失下评估。",
        },
        {
            "suffix": "E", "reasoning_structure_id": "FR06_CALIBRATION_DECISION_UTILITY",
            "prompt": "某模型 CRPS 更低但对一种高损失极端事件欠校准。如何比较它与基线的业务价值？写出概率校准和决策损失两个层次。",
            "reference": "先按事件与提前量检查可靠性、分辨率和校准；再用预先给定的行动成本、漏报损失和阈值计算期望决策损失；总体 CRPS 不能替代高损失事件效用。",
            "failures": ["用单一平均分决定部署", "没有行动成本或漏报损失"],
            "boundary": "业务价值依赖明确用户、阈值与损失函数。",
        },
    ],
    "FG06": [
        {
            "suffix": "A", "reasoning_structure_id": "FR06_SOURCE_MATURITY_LEAKAGE",
            "prompt": "把官方案例证据、独立盲评、预注册实验复现按成熟度排序，并指出评价泄漏会怎样制造虚高结果。",
            "reference": "官方案例只能提供早期可行性线索；独立盲评减少选择和声誉偏差；预注册实验复现提供更强效度；若评委看到模型身份、参考假设或结果，排序与新颖性评分会虚高。",
            "failures": ["把官方博客当独立复现", "未解释泄漏通道"],
            "boundary": "证据成熟度排序不否认案例价值，但限制可支持主张。",
        },
        {
            "suffix": "E", "reasoning_structure_id": "FR01_BLINDED_BASELINE_AUDIT",
            "prompt": "设计一个比较多智能体系统、单智能体与人类专家的盲化基线审计，写明材料截止时间、评分时点和失败判据。",
            "reference": "三组应共享相同文献截止时间和资源预算；在实验结果揭晓前由盲评者评分新颖性、可检验性和依据真实性；若有效假设率不优于基线或虚构依据率超阈值则失败。",
            "failures": ["不同组资源不等", "结果揭晓后再评假设质量"],
            "boundary": "该审计只比较设定任务与预算下的增益。",
        },
    ],
    "FG07": [
        {
            "suffix": "A", "reasoning_structure_id": "FR07_CONSTRUCT_VALIDITY_CROSS_LEVEL",
            "prompt": "把‘刺激诱导可塑性指标→学习样行为→意识’分成三个构念层级，指出材料直接触及哪一级、不能跨到哪一级。",
            "reference": "材料直接触及细胞与网络可塑性代理；只有任务依赖、保持和可逆干预等证据才能更接近学习样构念；这些仍不能推出主观体验或意识。",
            "failures": ["看到可塑性就宣称意识", "没有区分代理与构念"],
            "boundary": "结论必须停留在所测细胞和网络指标。",
        },
        {
            "suffix": "E", "reasoning_structure_id": "FR02_INTERVENTION_PROXY_TEST",
            "prompt": "设计一个区分‘真正的刺激特异可塑性’与‘一般兴奋性漂移’的干预：写出三组对照和相反预测。",
            "reference": "设置无刺激、随机刺激和阻断已知可塑性通路三组；真正可塑性应具刺激特异性、时间保持并被机制阻断削弱，一般漂移则不随配对规则或阻断呈相同模式。",
            "failures": ["只有前后比较", "没有机制阻断或随机刺激对照"],
            "boundary": "实验最多支持学习样可塑性机制，不支持意识结论。",
        },
    ],
    "FG08": [
        {
            "suffix": "A", "reasoning_structure_id": "FR08_MEASUREMENT_PIPELINE_ERROR",
            "prompt": "画出‘成像选区→分割/配准→捕获测序→空间解释’误差链，并为每一段给一个可观测质量指标。",
            "reference": "选区需盲化或重复抽样，分割/配准需位置误差，捕获需效率与污染率，解释需与正交原位测量一致；任一环节误差都可伪造亚细胞富集。",
            "failures": ["只报测序深度", "忽略分割与配准"],
            "boundary": "定位相关不自动证明 RNA 定位的功能因果。",
        },
        {
            "suffix": "E", "reasoning_structure_id": "FR03_ORTHOGONAL_RECOVERY",
            "prompt": "若某 RNA 被报告在线粒体附近富集，设计一种正交验证并预先写出会撤回‘亚细胞定位’结论的结果。",
            "reference": "可用带已知区室标记的 smFISH 或等价原位方法，并加入 spike-in 与分割扰动；若定位一致性低、污染高或结论随配准参数改变，则撤回稳健定位主张。",
            "failures": ["用同一测量流程自证", "没有撤回条件"],
            "boundary": "正交定位验证仍不能单独证明功能作用。",
        },
    ],
    "FG09": [
        {
            "suffix": "A", "reasoning_structure_id": "FR09_TRANSLATIONAL_EVIDENCE_LADDER",
            "prompt": "把体外编辑、体内递送与组织效应、长期安全、临床收益排成证据阶梯；每级写一个不可由前一级替代的读出。",
            "reference": "体外级看目标编辑与副产物；体内级看组织分布、递送和功能；长期安全看免疫、毒性与脱靶；临床级看患者结局与比较对照；低一级不能替代高一级。",
            "failures": ["把细胞编辑率当临床疗效", "遗漏递送或长期安全"],
            "boundary": "结论必须限定到已完成的验证层级。",
        },
        {
            "suffix": "E", "reasoning_structure_id": "FR04_SAFETY_EFFICACY_TRADEOFF",
            "prompt": "若提高编辑器剂量使目标编辑率上升但组织外编辑也上升，怎样预先定义‘不值得继续外推’的联合失败判据？",
            "reference": "需同时预注册最低有效编辑/功能阈值与关键组织脱靶、毒性或免疫上限；任何安全上限被突破都不能被平均编辑率抵消；应报告剂量—获益—风险曲线。",
            "failures": ["只优化目标编辑率", "用平均值掩盖关键组织风险"],
            "boundary": "联合判据只适用于指定递送系统、组织和疾病模型。",
        },
    ],
    "FG10": [
        {
            "suffix": "A", "reasoning_structure_id": "FR10_PHYSICAL_LIMITS_OOD",
            "prompt": "区分‘跨网格分辨率运行’与‘跨参数、边界条件或方程族泛化’，并给出守恒和长期稳定两个额外判据。",
            "reference": "跨网格可来自函数空间参数化但不保证 OOD；应在未见参数/边界或方程族上比较可靠数值解，并检查守恒残差与长时间滚动误差是否有界。",
            "failures": ["把跨分辨率等同任意 PDE 泛化", "只报单步点误差"],
            "boundary": "泛化结论限于明确测试的方程、参数与边界域。",
        },
        {
            "suffix": "E", "reasoning_structure_id": "FR05_ABLATION_OOD_STRESS",
            "prompt": "设计一个消融，区分性能来自 Fourier 算子结构还是训练分布覆盖；写出训练/测试切分与会削弱结构性主张的结果。",
            "reference": "保持数据量与优化相同，比较 FNO 与参数量匹配基线，并按参数或边界条件做真正 OOD 切分；若优势仅在随机同分布切分存在、OOD 或守恒读出消失，则不能把优势归因于通用算子偏置。",
            "failures": ["随机切分造成泄漏", "基线资源不匹配"],
            "boundary": "消融支持的是特定架构归因，不是所有 PDE 的普遍优越性。",
        },
    ],
}


FRONTIER_VALIDATION_SOURCES = [
    {
        "id": "FV01", "concept": "RFdiffusion 蛋白结构与功能设计", "discipline": "蛋白质设计",
        "source_title": "De novo design of protein structure and function with RFdiffusion",
        "source_url": "https://www.nature.com/articles/s41586-023-06415-8", "source_date": "2023-07-11",
        "brief": "论文用扩散模型生成蛋白骨架并通过结构预测与一部分实验表征验证设计；计算可设计性、获得正确折叠和实现目标结合或催化功能仍是不同层级。",
        "reasoning_structure_id": "FR01_CLAIM_EVIDENCE_MAP",
        "question": "为一个声称具有目标结合功能的设计写出最小正交验证链：对照、结构读出、功能读出和会撤回功能主张的结果。",
        "reference": "应与天然或较弱设计基线比较；用实验结构或等价正交方法验证折叠和界面；用目标特异结合读出验证功能；若折叠正确但无目标结合，只保留结构设计成功而撤回功能主张。",
        "failures": ["只用结构预测自证", "把正确折叠当目标功能"],
        "boundary": "结构与功能必须分别验证，且结论限于所测蛋白和条件。",
    },
    {
        "id": "FV02", "concept": "scGPT 单细胞基础模型", "discipline": "单细胞组学",
        "source_title": "scGPT: toward building a foundation model for single-cell multi-omics using generative AI",
        "source_url": "https://www.nature.com/articles/s41592-024-02201-0", "source_date": "2024-02-26",
        "brief": "scGPT 在超过三千万细胞上预训练，并报告细胞类型注释、批次整合、扰动响应和基因网络等下游任务；下游预测和注意力图不自动等于真实调控机制。",
        "reasoning_structure_id": "FR02_PREDICTION_MECHANISM_GAP",
        "question": "模型提出一条新的基因调控边。设计一个因果验证，写出阴性对照、干预、直接读出和失败判据。",
        "reference": "在相关细胞类型中扰动候选调控基因或位点，设非靶向与匹配阴性对照，读取目标基因及中介染色质变化；若效应方向不复现或不优于对照，则撤回因果调控边主张。",
        "failures": ["用注意力权重当因果证据", "没有干预或阴性对照"],
        "boundary": "单一细胞体系的干预只支持该背景下的调控关系。",
    },
    {
        "id": "FV03", "concept": "MatterGen 无机材料逆向设计", "discipline": "材料科学",
        "source_title": "A generative model for inorganic materials design",
        "source_url": "https://www.nature.com/articles/s41586-025-08628-5", "source_date": "2025-01-16",
        "brief": "MatterGen 生成满足稳定性与属性约束的晶体候选，并合成一个生成材料作概念验证；DFT 稳定、可合成、目标性质和可制造性仍需逐级验证。",
        "reasoning_structure_id": "FR03_GENERATION_VALIDITY_LADDER",
        "question": "为新生成晶体建立从 DFT 候选到可用材料的四级验证阶梯，并说明单个合成成功不能推出什么。",
        "reference": "依次验证更高精度计算与竞争相、可重复合成和相纯度、实测目标性质、加工与工作条件稳定性；单个样品成功不能推出候选总体命中率或工业可制造性。",
        "failures": ["把 DFT 稳定当可合成", "用单个成功样品外推总体"],
        "boundary": "每一级只支持对应性质、样品与工艺条件。",
    },
    {
        "id": "FV04", "concept": "串接玻色猫码量子纠错", "discipline": "量子信息",
        "source_title": "Hardware-efficient quantum error correction via concatenated bosonic qubits",
        "source_url": "https://www.nature.com/articles/s41586-025-08642-7", "source_date": "2025-02-26",
        "brief": "实验将噪声偏置猫量子比特与距离5重复码串接，报告相位翻转纠错的低阈值行为及有限码距逻辑存储误差；不可纠正位翻转仍给出误差下限。",
        "reasoning_structure_id": "FR04_SCALING_EXTRAPOLATION",
        "question": "材料同时报告‘低于阈值’和‘重复码没有真正渐近阈值’。解释二者如何不矛盾，并给出验证可用扩展区间的实验读出。",
        "reference": "相位翻转子通道可在所测范围随码距受抑，但不可纠正位翻转造成最终下限；应测多个码距下位翻转与相位翻转分项、总逻辑错误及最优光子数区间，识别增码距何时不再获益。",
        "failures": ["忽略不可纠正位翻转", "把有限区间改进说成渐近指数抑制"],
        "boundary": "扩展结论限于噪声偏置、门和码距的实测范围。",
    },
    {
        "id": "FV05", "concept": "NeuralGCM 混合天气与气候模拟", "discipline": "地球系统科学",
        "source_title": "Neural general circulation models for weather and climate",
        "source_url": "https://www.nature.com/articles/s41586-024-07744-y", "source_date": "2024-07-22",
        "brief": "NeuralGCM 把可微动力学求解器与学习组件结合，在天气、集合和规定海温的多年气候任务上报告竞争性能，但论文明确模型不能外推到显著不同的未来气候。",
        "reasoning_structure_id": "FR05_BENCHMARK_DEPLOYMENT_GAP",
        "question": "设计一个检验未来气候外推的压力测试，写出时间切分、物理读出、比较基线和失败判据。",
        "reference": "按气候状态或强迫情景做分布外切分，与物理 GCM 和持久性/统计基线比较；读取能量水分收支、极端事件和长期漂移；若守恒残差或气候统计在新强迫下系统失真，则外推失败。",
        "failures": ["用历史随机切分代替未来气候", "只报短期 RMSE"],
        "boundary": "压力测试只能支持明确强迫与状态范围内的气候外推。",
    },
    {
        "id": "FV06", "concept": "实时语音神经假体", "discipline": "神经工程",
        "source_title": "A high-performance neuroprosthesis for speech decoding and avatar control",
        "source_url": "https://www.nature.com/articles/s41586-023-06443-4", "source_date": "2023-08-23",
        "brief": "研究在一名重度瘫痪参与者上用高密度皮层表面记录实时解码文本、语音和面部动画；高性能单例展示可行性，但个体、植入稳定性和长期使用外部效度未定。",
        "reasoning_structure_id": "FR06_SOURCE_MATURITY_LEAKAGE",
        "question": "把‘单参与者可行性’推进到‘可推广临床技术’需要哪三层独立证据？为每层写一个防止选择偏差的设计。",
        "reference": "需要同一参与者跨月稳定性、预先纳入的多参与者外部验证、以及与现有辅助沟通的临床效用比较；可用固定测试集、连续入组和盲化终点评估减少选择偏差。",
        "failures": ["把单例速度外推所有患者", "没有独立测试或连续入组"],
        "boundary": "可行性、外部效度与临床效用须分层报告。",
    },
    {
        "id": "FV07", "concept": "基因编辑猪肾异种移植", "discipline": "移植医学",
        "source_title": "Design and testing of a humanized porcine donor for xenotransplantation",
        "source_url": "https://www.nature.com/articles/s41586-023-06594-4", "source_date": "2023-10-11",
        "brief": "论文构建含多类基因编辑的猪供体，并在非人灵长类肾移植中报告较长生存与免疫相容性证据；模型结果到人体长期安全与获益仍需临床层验证。",
        "reasoning_structure_id": "FR07_CONSTRUCT_VALIDITY_CROSS_LEVEL",
        "question": "列出从非人灵长类移植生存到人体长期临床获益的三个跨层缺口，并为每个缺口给出相匹配的证据。",
        "reference": "物种免疫差异需人体免疫与排斥监测，实验免疫抑制和临床方案差异需可实施方案比较，器官生存需长期功能、感染与患者结局；动物生存时间不能直接替代人体净获益。",
        "failures": ["把动物生存等同人体疗效", "遗漏免疫抑制与感染代价"],
        "boundary": "临床外推必须保留物种、方案和随访时间边界。",
    },
    {
        "id": "FV08", "concept": "JWST 岩质系外行星透射光谱", "discipline": "天体物理",
        "source_title": "A JWST transmission spectrum of the nearby Earth-sized exoplanet LHS 475 b",
        "source_url": "https://www.nature.com/articles/s41550-023-02064-z", "source_date": "2023-08-14",
        "brief": "两次凌星观测得到近似无特征的透射光谱，排除富氢和无云纯甲烷大气，但高云、稀薄大气或几乎无大气仍相容。",
        "reasoning_structure_id": "FR08_MEASUREMENT_PIPELINE_ERROR",
        "question": "为什么‘无特征’不是‘无大气’的同义词？提出一个新增波段或观测几何，并写出它能区分的两个替代解释。",
        "reference": "无特征可能来自高云遮蔽、低尺度高度稀薄大气或无大气；可增加中红外次食或更宽波段多次透射，利用热辐射或分子吸收区分云顶、成分和无大气情形；仍需处理恒星污染与仪器系统误差。",
        "failures": ["把未检测到特征当不存在", "没有明确替代模型"],
        "boundary": "新增观测只能削弱或区分具体大气模型，不能单次证明生命。",
    },
    {
        "id": "FV09", "concept": "空气中可扩展制备的钙钛矿—硅叠层电池", "discipline": "光伏材料",
        "source_title": "Solvent engineering for scalable fabrication of perovskite/silicon tandem solar cells in air",
        "source_url": "https://www.nature.com/articles/s41467-024-49351-5", "source_date": "2024-06-08",
        "brief": "研究通过溶剂工程在空气中制备叠层器件，报告小面积认证效率、16平方厘米器件与数百小时运行稳定性；实验室扩大面积仍不等于量产良率和多年户外寿命。",
        "reasoning_structure_id": "FR09_TRANSLATIONAL_EVIDENCE_LADDER",
        "question": "把该结果推进到商业组件需补哪四级证据？至少包含面积/良率、加速老化、户外运行和制造成本。",
        "reference": "需在组件面积上报告批次良率与均匀性，按标准协议做湿热和热循环，进行跨季节户外能量产出与衰减，评估材料、节拍和封装成本；单个冠军器件和数百小时测试不能替代这些层级。",
        "failures": ["只追求冠军效率", "把短时稳定当多年寿命"],
        "boundary": "商业化主张必须绑定组件尺度、标准老化与量产统计。",
    },
    {
        "id": "FV10", "concept": "GNoME 晶体稳定性筛选", "discipline": "计算材料学",
        "source_title": "Scaling deep learning for materials discovery",
        "source_url": "https://www.nature.com/articles/s41586-023-06735-9", "source_date": "2023-11-29",
        "brief": "GNoME 以图网络和 DFT 主动学习扩展计算稳定晶体候选，并报告部分独立实验实现；凸包稳定性仍受参考相、泛函、动力学与合成路径限制。",
        "reasoning_structure_id": "FR10_PHYSICAL_LIMITS_OOD",
        "question": "构造一个会让‘DFT 凸包稳定→实验可合成’外推失败的机制链，并为每一环给出可观测检查。",
        "reference": "遗漏竞争相或泛函误差可改变凸包位置，声子/有限温度可揭示动力学或热力学不稳定，成核势垒与前驱体路径可阻止合成；分别可用更高精度复算、声子和自由能、以及合成路径与相鉴定检查。",
        "failures": ["把0 K DFT稳定当可合成", "没有竞争相或动力学检查"],
        "boundary": "计算稳定、动态稳定与可合成是相邻但不同的主张。",
    },
]


def hard_base(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "suite": "zhili_foundational_hard_v2",
        "structure_id": item["structure_id"],
        "structure_group": item["title"],
        "discipline": item["discipline"],
        "difficulty": "advanced",
        "curriculum_anchor": [item["discipline"], "基础学科高阶推理"],
        "agent_capability": ["严谨推导", "条件辨析", "反例与证据边界"],
        "rule_target": item["rule"],
        "source": "curated-foundational-hard-v2",
    }


def rubric(
    reference: str,
    failures: list[str],
    boundary: str,
    *,
    overcautious: str = "",
    inference_links: list[str] | None = None,
    acceptable_alternatives: list[str] | None = None,
    minimum_required_claims: int | None = None,
) -> dict[str, Any]:
    claims = [part.strip() for part in reference.replace("。", "；").split("；") if part.strip()]
    payload = {
        "required_claims": claims,
        "required_inference_links": inference_links or [
            f"必须说明题设或材料如何支持：{claim}" for claim in claims
        ],
        "acceptable_alternatives": acceptable_alternatives or [
            "允许使用与参考要点逻辑等价的定理、构造、反例或条件分析，但不得改变关键前提与结论极性。"
        ],
        "fatal_errors": failures,
        "overcautious_failure": overcautious,
        "minimum_boundary_statement": boundary,
    }
    if minimum_required_claims is not None:
        payload["minimum_required_claims"] = minimum_required_claims
    return payload


def build_hard() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    development: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for item in HARD_GROUPS:
        for variant, question, reference, failures in item["development"]:
            development.append({
                **hard_base(item), "id": f"ZHF-{item['structure_id']}-{variant}",
                "split": "development", "variant": variant, "question": question,
                "reference": reference, "common_failures": failures, "frozen": False,
                "scoring_rubric": rubric(
                    reference, failures, item["rule"],
                    overcautious="若题设已给出完成参考推理所需的条件，仍只回答信息不足或拒绝推导，应扣分。",
                ),
            })
        for index, (question, reference) in enumerate(item["validation"], 1):
            validation.append({
                **hard_base(item), "id": f"ZHV-{item['structure_id']}-{index}",
                "split": "transfer_validation", "variant": "V", "question": question,
                "reference": reference, "common_failures": ["只迁移术语，没有迁移条件结构"], "frozen": True,
                "scoring_rubric": rubric(
                    reference,
                    ["只迁移术语，没有迁移条件结构"],
                    item["rule"],
                    overcautious="若题设条件充分支持参考结论，仍一律回答信息不足，应扣分。",
                    minimum_required_claims=3 if item["structure_id"] == "HF08" and index == 1 else None,
                ),
            })
    challenge = []
    for hid, title, question, reference, failures, boundary in HARD_CHALLENGE:
        row = {
            "id": f"ZHC-{hid}", "suite": "zhili_foundational_hard_v2",
            "split": "author_visible_challenge", "structure_id": hid,
            "structure_group": title, "discipline": "跨基础学科", "difficulty": "advanced",
            "variant": "H", "curriculum_anchor": ["跨基础学科"],
            "agent_capability": ["新结构识别"], "question": question, "reference": reference,
            "common_failures": failures, "rule_target": "冻结挑战，不参与规则修改。",
            "scoring_rubric": rubric(
                reference,
                failures,
                boundary,
                overcautious="若题设要求构造、计算或肯定推导，不能用笼统的‘信息不足’替代作答。",
            ),
            "source": "curated-foundational-hard-v2", "frozen": True,
        }
        if hid in HARD_ASSERTION_SPECS:
            row["hard_assertion"] = HARD_ASSERTION_SPECS[hid]
        challenge.append(row)
    return development, validation, challenge


def build_frontier() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    development: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for source in FRONTIER_SOURCES:
        audit = FRONTIER_SOURCE_AUDIT[source["id"]]
        common = {
            "suite": "zhili_frontier_guided_reading_v1",
            "concept_id": source["id"], "structure_group": source["concept"],
            "discipline": source["discipline"], "difficulty": "frontier-guided-reading",
            "curriculum_anchor": [source["discipline"], "论文领读"],
            "source_title": source["source_title"], "source_url": source["source_url"],
            "source_date": source["source_date"], "reading_brief": source["brief"],
            "source_claims": audit["source_claims"],
            "reported_evidence": audit["reported_evidence"],
            "curator_inference": audit["curator_inference"],
            "known_unvalidated_links": audit["known_unvalidated_links"],
            "source_type": audit["source_type"],
            "evidence_maturity": "early_official_report" if audit["source_type"] == "official_research_blog" else "peer_reviewed_primary",
            "claim_locator": "abstract-and-main-results",
            "source": "primary-source-guided-reading-v1",
        }
        for task in FRONTIER_DEV_TASKS[source["id"]]:
            reference = task["reference"]
            family_id = task["reasoning_structure_id"].split("_", 1)[0]
            development.append({
                **common,
                "id": f"ZFG-{source['id']}-{task['suffix']}",
                "split": "development",
                "variant": task["suffix"],
                "structure_id": family_id,
                "reasoning_family_id": family_id,
                "reasoning_structure_id": task["reasoning_structure_id"],
                "transfer_mode": "development",
                "frozen": False,
                "question": f"领读材料（摘要转述）：{source['brief']}\n{task['prompt']}",
                "reference": reference,
                "common_failures": task["failures"],
                "rule_target": task["boundary"],
                "scoring_rubric": rubric(
                    reference,
                    task["failures"],
                    task["boundary"],
                    overcautious="若材料已给出完成任务所需信息，不能用笼统的‘仍需研究’代替具体分析。",
                ),
            })

    for source in FRONTIER_VALIDATION_SOURCES:
        reference = source["reference"]
        family_id = source["reasoning_structure_id"].split("_", 1)[0]
        validation.append({
            "suite": "zhili_frontier_guided_reading_v1",
            "id": f"ZFV-{source['id']}-V",
            "concept_id": source["id"],
            "split": "transfer_validation",
            "variant": "V",
            "structure_id": family_id,
            "reasoning_family_id": family_id,
            "reasoning_structure_id": source["reasoning_structure_id"],
            "structure_group": source["concept"],
            "transfer_mode": "different_source",
            "discipline": source["discipline"],
            "difficulty": "frontier-guided-reading",
            "curriculum_anchor": [source["discipline"], "论文领读"],
            "source_title": source["source_title"],
            "source_url": source["source_url"],
            "source_date": source["source_date"],
            "reading_brief": source["brief"],
            "source_claims": [source["brief"].split("；", 1)[0]],
            "reported_evidence": [source["brief"].split("；", 1)[0]],
            "curator_inference": [source["boundary"]],
            "known_unvalidated_links": [source["boundary"]],
            "source_type": "peer_reviewed_article",
            "evidence_maturity": "peer_reviewed_primary",
            "claim_locator": "abstract-and-main-results",
            "source": "primary-source-guided-reading-v1",
            "frozen": True,
            "question": f"领读材料（摘要转述）：{source['brief']}\n{source['question']}",
            "reference": reference,
            "common_failures": source["failures"],
            "rule_target": "冻结跨来源同结构迁移；不得据此修改规则。",
            "scoring_rubric": rubric(
                reference,
                source["failures"],
                source["boundary"],
                overcautious="若材料足以支持有边界的设计或判断，不能一律拒绝。",
            ),
        })
    return development, validation


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_manifest(path: Path, suite: str, counts: dict[str, int], note: str) -> None:
    payload = {
        "suite": suite,
        "revision": 2 if suite in {"zhili_foundational_hard_v2", "zhili_frontier_guided_reading_v1"} else 1,
        "counts": counts,
        "policy": {
            "development": "可用于归因和修改通用规则。",
            "transfer_validation": "冻结后运行；不得根据答案修改规则。",
            "author_visible_challenge": "仅作作者可见迁移快照，不声称严格盲测。",
        },
        "design_note": note,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    hard_dev, hard_val, hard_challenge = build_hard()
    write_jsonl(HARD_PACK / "development_30.jsonl", hard_dev)
    write_jsonl(HARD_PACK / "transfer_validation_20.jsonl", hard_val)
    write_jsonl(HARD_PACK / "author_visible_challenge_10.jsonl", hard_challenge)
    write_manifest(
        HARD_PACK / "manifest.json", "zhili_foundational_hard_v2",
        {"development": len(hard_dev), "transfer_validation": len(hard_val), "author_visible_challenge": len(hard_challenge)},
        "10 个开发推理结构，每组 3 个异质变体与 2 个冻结迁移；另含 10 个构造、肯定推导、计算和纠错混合的新结构挑战。",
    )

    frontier_dev, frontier_val = build_frontier()
    write_jsonl(FRONTIER_PACK / "development_20.jsonl", frontier_dev)
    write_jsonl(FRONTIER_PACK / "transfer_validation_10.jsonl", frontier_val)
    write_manifest(
        FRONTIER_PACK / "manifest.json", "zhili_frontier_guided_reading_v1",
        {"development": len(frontier_dev), "transfer_validation": len(frontier_val)},
        "开发集含 10 个一手来源概念与 20 个细粒度任务，归入 10 个推理家族；冻结集使用 10 个完全不同的一手来源概念，测试跨来源同结构迁移。",
    )
    print(json.dumps({"hard": [len(hard_dev), len(hard_val), len(hard_challenge)], "frontier": [len(frontier_dev), len(frontier_val)]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
