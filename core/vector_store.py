import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # Optional: the main app uses the lightweight local index.
    HuggingFaceEmbeddings = None


# 获取本地 HuggingFace Embedding 模型（BAAI/bge-small-zh-v1.5）
def get_local_embeddings():
    if HuggingFaceEmbeddings is None:
        raise RuntimeError(
            "BGE/FAISS 兼容模块需要额外安装：pip install langchain-huggingface sentence-transformers"
        )
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


# 1. 加载本地 PDF 目录中的所有教材文件
def load_pdf_textbooks(pdf_dir: str):
    loader = DirectoryLoader(
        pdf_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()
    print(f"成功加载 {len(docs)} 页 PDF 内容")
    return docs


# 2. 文本切块
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "；", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"切分出 {len(chunks)} 个知识块")
    return chunks


# 3. 本地构建并保存 FAISS 向量库（完全离线，不消耗 API 额度）
def build_and_save_vectorstore(pdf_dir: str, save_path: str):
    docs = load_pdf_textbooks(pdf_dir)
    chunks = split_documents(docs)

    print("正在加载本地 BGE 向量模型...")
    embeddings = get_local_embeddings()

    print("开始本地生成向量索引（使用 CPU，请稍候 1~2 分钟）...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(save_path)
    print(f"FAISS 向量库已成功保存至: {save_path}")
    return vectorstore


# 4. 加载已有向量库
def load_vectorstore(save_path: str):
    embeddings = get_local_embeddings()
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
