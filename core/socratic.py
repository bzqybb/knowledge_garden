from typing import Dict, List

from core.llm import LLMError, chat_json


# 1. 碎片兴趣存储与隐性联系挖掘
def discover_cross_domain_link(concept: str, textbook_ref: str, user_interests: List[str], api_key: str) -> Dict:
    """
    挖掘【硬核专业概念】与【用户碎片兴趣】之间的隐藏底层逻辑关联
    """
    prompt = f"""你是一位富有哲思与创意灵感的“知识园丁”导师。

硬核专业概念：【{concept}】
课本基础背景：{textbook_ref[:300]}
用户收藏的碎片兴趣/生活灵感：{user_interests}

请在“专业概念”与“碎片兴趣”之间找出一个深层的**底层同构关系**（例如：信号滤波与情绪管理中的“噪点剔除”、热力学熵增与文章结构的“紊乱度”）。

请生成包含以下两部分的创意对照：
1. 【隐性隐喻】：用一句话揭示这个跨界碰撞的核心火花。
2. 【苏格拉底追问】：提出 1~2 个不给直接答案、但极具启发性的追问，引导用户主动思考。

按 JSON 格式输出：
{{
    "metaphor": "跨界隐喻内容",
    "socratic_questions": ["追问1", "追问2"]
}}
"""

    try:
        result = chat_json("你是知识园丁，只返回 JSON。", prompt)
        if result:
            return result
    except LLMError:
        pass
    return {
            "metaphor": f"【{concept}】的底层结构与你关注的兴趣点有着异曲同工的动态平衡之美。",
            "socratic_questions": ["如果把这个专业公式中的变量替换为你生活中的一个要素，它会是什么？"]
        }
