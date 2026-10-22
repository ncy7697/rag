import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ---------------------------------------------------------
# 1. 설정 및 임베딩/LLM 초기화
# ---------------------------------------------------------
PERSIST_DIRECTORY = "./chroma_db"

# [임베딩 선택]
# 1) 로컬 Ollama: nomic-embed-text
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2) 로컬 HuggingFace (sentence-transformers 사용 시):
# from langchain_huggingface import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3) 클라우드 OpenAI (사용 시 주석 해제 및 API Key 설정):
# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# [LLM 선택 - 로컬 Ollama]
llm = ChatOllama(
    model="llama3.2",  # 또는 mistral 등
    temperature=0.1,
)

# ---------------------------------------------------------
# 2. 문서 로드 및 벡터 DB(Chroma) 구축
# ---------------------------------------------------------
# 테스트용 샘플 텍스트 파일 생성 (처음 실행 시)
sample_file_path = "sample_knowledge.txt"
if not os.path.exists(sample_file_path):
    with open(sample_file_path, "w", encoding="utf-8") as f:
        f.write(
            "LangChain은 LLM을 활용한 애플리케이션 개발을 위한 프레임워크입니다.\n"
            "Chroma는 오픈소스 벡터 데이터베이스로 로컬 환경에서 임베딩을 저장하고 검색하는 데 적합합니다.\n"
            "Ollama를 사용하면 로컬 PC에서도 Llama 3.2나 Mistral 같은 강력한 오픈소스 LLM을 구동할 수 있습니다."
        )

print("1. 문서 로드 및 청크 분할 중...")
loader = TextLoader(sample_file_path, encoding="utf-8")
docs = loader.load()

# 텍스트 청크 분할 (Chunking)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50
)
splits = text_splitter.split_documents(docs)

print("2. ChromaDB 벡터 스토어 생성 및 저장 중...")
vectorstore = Chroma.from_documents(
    documents=splits, embedding=embeddings, persist_directory=PERSIST_DIRECTORY
)

# Retriever 생성 (상위 2개의 관련 문서를 검색)
retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 2}
)

# ---------------------------------------------------------
# 3. RAG 프롬프트 및 체인(Chain) 구성
# ---------------------------------------------------------
template = """다음의 검색된 참고 문서를 바탕으로 질문에 답변해 주세요.
만약 참고 문서에서 답을 찾을 수 없다면 "모르겠습니다"라고 답변해 주세요.

[참고 문서]
{context}

[질문]
{question}

[답변]:"""

prompt = ChatPromptTemplate.from_template(template)


# 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# LCEL(LangChain Expression Language)을 이용한 RAG 파이프라인 구축
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---------------------------------------------------------
# 4. RAG 실행 테스트
# ---------------------------------------------------------
if __name__ == "__main__":
    # query = "Ollama를 사용하면 어떤 장점이 있나요?"
    query = "성남의 중원초등학교에 대해서 알아?"
    print(f"\n[질문]: {query}\n")

    print("[답변 생성 중...]")
    response = rag_chain.invoke(query)

    print(f"\n[AI 답변]:\n{response}")