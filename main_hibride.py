import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


# ********* Hibride Searching (하이브리드 검색 및 태그 필터링) 적용


app = FastAPI(
    title="Hybrid Tagged RAG API",
    description="FastAPI + Chroma + BM25 + Ollama (하이브리드 검색 및 태그 필터링)",
    version="2.0"
)

PERSIST_DIRECTORY = "./chroma_db_hybrid"

# 1. 임베딩 및 LLM 초기화
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2", temperature=0.1)

# 2. ChromaDB 벡터 스토어 로드
vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

# ---------------------------------------------------------
# Pydantic 모델 정의
# ---------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str
    tag: Optional[str] = None  # 특정 태그 지정 시 해당 문서들만 대상로 검색
    k: Optional[int] = 3

class AnswerResponse(BaseModel):
    question: str
    answer: str
    used_tag: Optional[str]
    sources: List[str]

# ---------------------------------------------------------
# API 엔드포인트
# ---------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Hybrid Tagged RAG API Server is running!"}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    tag: Optional[str] = Form(None),      # 예: "tech", "hr", "finance" 등 태그
    category: Optional[str] = Form(None)  # 예: "manual", "report" 등 카테고리
):
    """
    파일을 업로드하고 태그 및 메타데이터를 함께 부여하여 ChromaDB에 적재합니다.
    """
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix in [".txt", ".md"]:
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식입니다: {suffix}")
        
        docs = loader.load()
        
        # 메타데이터에 파일명, 태그, 카테고리 주입
        for doc in docs:
            doc.metadata["source"] = file.filename
            if tag:
                doc.metadata["tag"] = tag
            if category:
                doc.metadata["category"] = category
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        vectorstore.add_documents(splits)
        os.unlink(tmp_path)
        
        return {
            "status": "success",
            "message": f"'{file.filename}' 파일이 태그([ {tag} ])와 함께 성공적으로 색인되었습니다.",
            "filename": file.filename,
            "tag": tag
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    지정한 태그 기반으로 문서를 필터링한 뒤, BM25와 벡터 검색을 결합한 하이브리드 검색을 수행합니다.
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="질문 내용이 비어 있습니다.")
        
        # 1. 태그 필터 조건 설정
        where_filter = {"tag": request.tag} if request.tag else None
        
        # 2. ChromaDB에서 조건에 맞는 문서 조각 가져오기 (BM25 빌드용)
        raw_data = vectorstore.get(where=where_filter)
        filtered_docs = []
        if raw_data and "documents" in raw_data and raw_data["documents"]:
            for content, meta in zip(raw_data["documents"], raw_data["metadatas"]):
                filtered_docs.append(Document(page_content=content, metadata=meta))
        
        if not filtered_docs:
            raise HTTPException(status_code=404, detail=f"태그([ {request.tag} ])에 해당하는 문서가 존재하지 않습니다.")

        # 3. 리트리버 구성
        # 벡터 검색 리트리버 (Chroma 자체 메타데이터 필터 적용)
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": request.k, **({"filter": where_filter} if where_filter else {})}
        )
        
        # BM25 키워드 검색 리트리버 (필터링된 문서 집합 기반 실시간 생성)
        bm25_retriever = BM25Retriever.from_documents(filtered_docs)
        bm25_retriever.k = request.k
        
        # 앙상블 리트리버 결합 (BM25 40%, 벡터 검색 60%)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )
        
        # 4. 프롬프트 및 체인 구성
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
        
        rag_chain = (
            {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # 5. 실행 및 출처 수집
        retrieved_docs = ensemble_retriever.invoke(request.question)
        sources = list(set([doc.metadata.get("source", "unknown") for doc in retrieved_docs]))
        answer = rag_chain.invoke(request.question)
        
        return AnswerResponse(
            question=request.question,
            answer=answer,
            used_tag=request.tag,
            sources=sources
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))