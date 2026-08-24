from langchain_core.tools import tool

from core.retrieval import search_notes
from core.storage import GardenStore

class TextbookRetrieverTool:
    def __init__(self, vectorstore_path: str, api_key: str):
        self.store = GardenStore()

    def get_tool(self):
        @tool
        def query_textbook_knowledge(concept: str) -> str:
            """
            输入前沿学术/技术概念，从本地教材知识库中检索最相关的基础课本知识点与公式段落。
            """
            docs = search_notes(self.store, concept, kinds={"textbook", "course", "concept"}, limit=3)
            context = "\n---\n".join([f"[来源: {d['title']}]\n{d['snippet']}" for d in docs])
            return context if context else "未在教材库中检索到直接匹配的基础知识。"

        return query_textbook_knowledge
