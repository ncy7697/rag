import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever

# [추가] Reranker (Contextual Compression) 관련 임포트
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank


# ********* Rerank 적용
# 1차 검색 (Parent-Child): 사용자가 질문하면 벡터 스토어에서 자식 조각 기준으로 대략 5개(k=5)의 후보 부모 문서를 건져 올립니다.
# 2차 정밀 재채점 (Reranker - FlashRank): 건져 올린 문서들과 사용자의 질문 간 실제 문맥 일치도를 크로스 앙코더 방식으로 
# 깊게 재분석하여 점수를 매긴 뒤 순위를 다시 매깁니다.   
# 최종 압축: 그중 가장 점수가 높은 상위 3개(top_n=3)만 남겨서 LLM에게 전달하므로, 관련 없는 정보(노이즈)로 인한 환각 현상이 획기적으로 줄어듭니다.   



app = FastAPI(
    title="Advanced Parent-Child RAG API with Reranker",
    description="FastAPI + Chroma + Parent-Child + FlashRank Reranker + Ollama",
    version="3.3"
)

PERSIST_DIRECTORY = "./chroma_db_parent_child"

# 1. 임베딩 및 LLM 초기화
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2", temperature=0.1)

# 2. ChromaDB 벡터 스토어 (자식 조각 저장용) 초기화
vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

# 3. 부모 문서 저장소 설정 (메모리 스토어)
parent_store = InMemoryStore()

# 4. 분할기 설정 (Parent: 1500자 / Child: 400자)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# 5. 기본 ParentDocumentRetriever 구성
base_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=parent_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 6. [추가] Reranker (FlashRank) 설정 및 압축 리트리버로 래핑
# top_n=3: 리랭킹 후 최종적으로 LLM에게 전달할 상위 문서 개수
compressor = FlashrankRerank(top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# ---------------------------------------------------------
# Pydantic 모델 정의
# ---------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str
    k: Optional[int] = 5  # Reranking을 위해 1차로 넉넉하게 가져올 후보 개수 (기본 5개)

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]

# ---------------------------------------------------------
# API 엔드포인트
# ---------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Advanced RAG API with Reranker is running!"}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: Optional[str] = Form(None)
):
    """
    파일을 업로드받아 자동으로 Parent-Child 구조로 쪼개어 인덱싱합니다.
    """
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        elif suffix in [".txt", ".md"]:
            try:
                loader = TextLoader(tmp_path, encoding="utf-8")
                docs = loader.load()
            except UnicodeDecodeError:
                loader = TextLoader(tmp_path, encoding="cp949")
                docs = loader.load()
        else:
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식입니다: {suffix}")
        
        for doc in docs:
            doc.metadata["source"] = file.filename
            if category:
                doc.metadata["category"] = category
        
        # 기본 base_retriever를 통해 문서 추가 수행
        base_retriever.add_documents(docs)
        os.unlink(tmp_path)
        
        return {
            "status": "success",
            "message": f"'{file.filename}' 파일이 Parent-Child + Reranker 파이프라인에 색인되었습니다.",
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    1차 검색(Parent-Child) -> 2차 정밀 재채점(Reranker) 과정을 거쳐 답변을 생성합니다.
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="질문 내용이 비어 있습니다.")
        
        # 1차로 가져올 후보 개수 설정 (Reranker가 정밀하게 거르기 위해 넉넉히 설정)
        base_retriever.search_kwargs = {"k": request.k}
        
        template = """다음의 참고 문서를 바탕으로 질문에 정확하게 답변해 주세요.
만약 참고 문서에서 답을 찾을 수 없다면 "참고 문서에서 답을 찾을 수 없습니다."라고 답변해 주세요.

[참고 문서]
{context}

[질문]
{question}

[답변]:"""

        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # RAG 체인 구성 (compression_retriever를 통해 검색 및 리랭킹 수행)
        rag_chain = (
            {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # 검색 및 리랭킹 실행
        retrieved_docs = compression_retriever.invoke(request.question)
        sources = list(set([doc.metadata.get("source", "unknown") for doc in retrieved_docs]))
        answer = rag_chain.invoke(request.question)
        
        return AnswerResponse(
            question=request.question,
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))