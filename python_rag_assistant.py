import os
import PyPDF2
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import tempfile
from dotenv import load_dotenv
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PythonRAGAssistant:
    def __init__(self, model_path=None):
        self.vector_store = None  # 后面存 FAISS 索引
        self.qa_chain = None  # 后面存 RetrievalQA 链条
        self.embeddings = None  # 后面存 HuggingFace 句向量模型
        self.documents = []  # 每本 PDF 的纯文本整段
        self.load_env()
        # 优先使用环境变量中的模型路径，如果没有则使用传入的model_path
        env_model_path = os.getenv("LOCAL_EMBEDDING_MODEL_PATH")
        self.initialize_embeddings(env_model_path if env_model_path else model_path)

    def load_env(self):
        """加载环境变量，包括离线模型路径"""
        load_dotenv()
        self.qwen_api_key = os.getenv("QWEN_API_KEY")
        self.local_embedding_model_path = os.getenv("LOCAL_EMBEDDING_MODEL_PATH")

        if not self.qwen_api_key:
            logger.warning("QWEN_API_KEY 环境变量未设置，将在Streamlit界面中提示用户输入")

        if self.local_embedding_model_path:
            logger.info(f"从环境变量获取到本地嵌入模型路径: {self.local_embedding_model_path}")
        else:
            logger.info("未在环境变量中找到本地嵌入模型路径(LOCAL_EMBEDDING_MODEL_PATH)")

    def initialize_embeddings(self, model_path=None):
        """初始化嵌入模型，优先使用本地模型，不依赖网络"""
        try:
            # 优先使用本地模型
            if model_path and os.path.exists(model_path):
                # 从本地文件加载模型
                self.embeddings = HuggingFaceEmbeddings(model_name=model_path)
                logger.info(f"✅ 成功从本地路径加载嵌入模型: {model_path}")
            else:
                # 检查是否提供了模型路径但路径不存在
                if model_path:
                    logger.error(f"❌ 本地模型路径不存在: {model_path}")
                    raise FileNotFoundError(f"本地模型路径不存在: {model_path}")

                # 如果没有提供有效的本地模型路径，提示用户配置
                logger.error("❌ 未配置有效的本地嵌入模型路径")
                raise ValueError(
                    "请配置本地嵌入模型路径。\n"
                    "1. 在.env文件中设置LOCAL_EMBEDDING_MODEL_PATH指向您的本地模型目录\n"
                    "2. 确保模型已正确下载到该目录"
                )

        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {str(e)}")
            raise

    def load_pdf_documents(self, pdf_files):
        """加载并处理PDF文档"""
        self.documents = []

        for pdf_file in pdf_files:
            try:
                # 创建临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(pdf_file.read())
                    tmp_file_path = tmp_file.name

                # 读取PDF内容
                pdf_reader = PyPDF2.PdfReader(tmp_file_path)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    extracted_text = page.extract_text()
                    if extracted_text:  # 只添加非空文本
                        text += extracted_text + "\n"

                # 清理临时文件
                os.unlink(tmp_file_path)

                if text:  # 只添加有内容的文档
                    self.documents.append(text)
                    logger.info(f"成功加载PDF文档: {pdf_file.name}，提取到{len(text)}个字符")
                else:
                    logger.warning(f"PDF文档{pdf_file.name}中未提取到文本内容")

            except Exception as e:
                logger.error(f"加载PDF文档失败 {pdf_file.name}: {str(e)}")
                raise

        return len(self.documents)

    def create_vector_store(self):
        """创建向量存储"""
        try:
            if not self.documents:
                raise ValueError("没有可处理的文档，请先加载PDF文件")

            # 分割文本
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n", "\n\n", "", " "]
            )

            all_splits = []
            for doc in self.documents:
                splits = text_splitter.split_text(doc)
                all_splits.extend(splits)

            # 创建向量存储
            self.vector_store = FAISS.from_texts(texts=all_splits, embedding=self.embeddings)
            logger.info(f"向量存储创建成功，包含 {len(all_splits)} 个文本块")

        except Exception as e:
            logger.error(f"向量存储创建失败: {str(e)}")
            raise

    def initialize_qa_chain(self, api_key=None, model_url=None):
        """初始化问答链，支持离线LLM模型"""
        if not self.vector_store:
            raise ValueError("向量存储未初始化，请先创建向量存储")

        # 如果未提供API Key，使用类实例中的API Key（从环境变量加载的）
        if not api_key:
            api_key = self.qwen_api_key
            if not api_key:
                raise ValueError("API Key未提供，请在.env文件中设置或在界面中输入")

        try:
            # 使用OpenAI兼容接口初始化模型，支持本地部署的模型
            llm_params = {
                "api_key": api_key,
                # 默认为通义千问模型，如果使用本地模型可在环境变量中配置
                "model": os.getenv("LLM_MODEL_NAME", "qwen-max")
            }

            # 优先使用环境变量中的模型URL，如果界面提供了则覆盖
            env_model_url = os.getenv("LLM_MODEL_URL")
            if model_url:
                llm_params["base_url"] = model_url
            elif env_model_url:
                llm_params["base_url"] = env_model_url
                logger.info(f"使用环境变量中的模型URL: {env_model_url}")
            else:
                # 默认使用通义千问官方API地址
                llm_params["base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
                logger.info("使用默认的通义千问官方API地址")

            # 初始化OpenAI兼容接口的模型
            llm = ChatOpenAI(**llm_params)

            # 创建检索问答链
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True
            )

            logger.info("问答链初始化成功")

        except Exception as e:
            logger.error(f"问答链初始化失败: {str(e)}")
            raise

    def get_answer(self, question):
        """获取问题的答案"""
        if not self.qa_chain:
            raise ValueError("问答链未初始化，请先初始化问答链")

        try:
            result = self.qa_chain.invoke({"query": question})

            # 提取相关信息
            answer = result["result"]
            source_documents = result["source_documents"]

            # 构建带有溯源的答案
            cited_answer = f"{answer}\n\n【知识来源】\n"
            for i, doc in enumerate(source_documents, 1):
                cited_answer += f"{i}. {doc.page_content[:100]}...\n"

            return cited_answer

        except Exception as e:
            logger.error(f"获取答案失败: {str(e)}")
            return f"抱歉，无法回答您的问题。错误信息: {str(e)}"


# 创建Streamlit界面
def create_streamlit_app():
    st.set_page_config(
        page_title="Python语法专属RAG助手",
        page_icon="🐍",
        layout="wide"
    )

    # 初始化RAG助手
    if "rag_assistant" not in st.session_state:
        try:
            st.session_state.rag_assistant = PythonRAGAssistant()
            st.success("✅ 成功初始化RAG助手，使用本地嵌入模型")
        except Exception as e:
            st.error(f"初始化RAG助手失败: {str(e)}")
            return

    # 从环境变量加载默认配置
    default_api_key = os.getenv("QWEN_API_KEY", "")
    default_model_url = os.getenv("LLM_MODEL_URL", "")
    local_embedding_path = os.getenv("LOCAL_EMBEDDING_MODEL_PATH", "")

    st.title("🐍 Python语法专属RAG助手")
    st.markdown("### 专为Python学习者/开发者打造的精准知识点问答工具")

    # 显示当前使用的模型信息
    with st.expander("🔍 模型信息", expanded=False):
        st.info(f"当前使用的嵌入模型路径: {local_embedding_path or '未配置'}")
        if default_model_url:
            st.info(f"当前使用的LLM模型URL: {default_model_url}")

    # 侧边栏 - 上传PDF文档和API Key设置
    with st.sidebar:
        st.header("设置")

        # API Key输入，默认值为从环境变量读取的值
        qwen_api_key = st.text_input(
            "请输入模型API Key",
            type="password",
            placeholder="sk-xxxxxxxxxxxxxxxx",
            value=default_api_key  # 默认使用环境变量中的值
        )

        # 添加模型URL输入，默认使用环境变量中的值
        qwen_model_url = st.text_input(
            "请输入模型URL",
            placeholder="例如: http://localhost:8000/v1 或官方API地址",
            value=default_model_url,
            help="可设置为本地部署的模型服务地址，如未设置将使用环境变量中的LLM_MODEL_URL或默认地址"
        )

        # PDF上传
        st.subheader("上传Python相关PDF文档")
        pdf_files = st.file_uploader(
            "支持多文件上传",
            type="pdf",
            accept_multiple_files=True,
            help="建议上传Python官方文档、Python核心编程等PDF"
        )

        # 处理按钮
        if st.button("处理文档"):
            # 如果界面中未输入API Key，但环境变量中有，则使用环境变量中的
            if not qwen_api_key and default_api_key:
                qwen_api_key = default_api_key

            if not qwen_api_key:
                st.error("请输入模型API Key")
            elif not pdf_files:
                st.error("请上传PDF文档")
            else:
                with st.spinner("正在处理文档，请稍候..."):
                    try:
                        # 加载文档
                        doc_count = st.session_state.rag_assistant.load_pdf_documents(pdf_files)

                        # 创建向量存储
                        st.session_state.rag_assistant.create_vector_store()

                        # 初始化问答链，传递模型URL
                        st.session_state.rag_assistant.initialize_qa_chain(qwen_api_key, qwen_model_url)

                        st.success(f"成功处理了 {doc_count} 个文档，可以开始提问了！")
                        st.session_state.ready = True
                    except Exception as e:
                        st.error(f"处理文档失败: {str(e)}")

    # 主界面 - 提问区域
    st.subheader("请输入您的问题")

    # 示例问题提示
    with st.expander("💡 查看示例问题"):
        st.markdown("- Python 装饰器怎么用？")
        st.markdown("- 列表推导式和生成器表达式的区别是什么？")
        st.markdown("- 什么是闭包？请举个例子说明")
        st.markdown("- 如何高效处理大型文件？")
        st.markdown("- Python中的GIL是什么？它有什么影响？")

    # 问题输入框
    question = st.text_input(
        "",
        placeholder="例如：Python装饰器怎么用？"
    )

    # 提问按钮
    if st.button("获取答案"):
        if not hasattr(st.session_state, "ready") or not st.session_state.ready:
            st.error("请先上传并处理PDF文档")
        elif not question:
            st.error("请输入您的问题")
        else:
            with st.spinner("正在生成答案，请稍候..."):
                try:
                    answer = st.session_state.rag_assistant.get_answer(question)

                    # 显示答案
                    st.markdown("### 📝 答案")
                    st.markdown(answer)

                except Exception as e:
                    st.error(f"获取答案失败: {str(e)}")

    # 页脚信息
    st.markdown("\n")
    st.markdown("---")
    st.markdown("📚 Python语法专属RAG助手 | 答案基于您上传的PDF文档")


if __name__ == "__main__":
    create_streamlit_app()
