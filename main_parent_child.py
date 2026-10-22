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
# [수정된 부분] 최신 LangChain 구조에 맞춘 임포트 경로
from langchain_classic.storage import LocalFileStore
from langchain_classic.retrievers import ParentDocumentRetriever


# ********* Parent-Child Chunking (계층형 청킹) 적용


app = FastAPI(
    title="Parent-Child RAG API",
    description="FastAPI + Chroma + Parent-Child Chunking + Ollama",
    version="3.0"
)

PERSIST_DIRECTORY = "./chroma_db_parent_child"
PARENT_STORE_DIRECTORY = "./parent_store"

# 1. 임베딩 및 LLM 초기화
# 역할: 텍스트를 숫자로 이루어진 벡터(Vector)로 변환해 주는 모델입니다.
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 역할: 검색된 참고 문서를 바탕으로 최종 답변을 생성하는 언어 모델입니다.
# temperature=0.1 설정: 값이 0에 가까울수록 창의적인 답변보다는 참고 문서에 기반한 사실적이고 일관된 답변을 생성하도록 통제합니다.
llm = ChatOllama(model="llama3.2", temperature=0.1)

# 2. ChromaDB 벡터 스토어 (자식 조각 저장용) 및 부모 문서 저장소 초기화
# 역할: 위에서 쪼개진 자식 조각들의 벡터 임베딩을 로컬 디렉토리에 영구 저장하고 유사도 검색을 수행합니다.
vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

# 부모 문서 영구 저장을 위한 파일 시스템 스토어
# 역할: 자식 조각이 가리키고 있는 원본 부모 문서(1500자 크기)의 원본 텍스트를 로컬 파일 시스템에 안전하게 저장하고 관리합니다.
parent_store = LocalFileStore(PARENT_STORE_DIRECTORY)

# 3. 분할기 설정 (Parent: 1500자 / Child: 400자)
# 의미: 문서를 1500자 단위의 큼직한 조각(Parent)으로 자르며, 조각 간에 150자씩 겹치게(Overlap) 하여 문맥이 중간에 끊기는 것을 방지합니다. 
# LLM에게 최종적으로 전달되어 답변의 근거가 되는 풍부한 맥락을 제공합니다.
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

# 의미: 부모 조각 내부를 다시 400자 크기의 아주 작은 조각(Child)으로 쪼갭니다. ChromaDB에서 사용자의 질문과 
# 가장 유사한 위치를 정밀하게 찾아내는(검색용) 타겟이 됩니다.
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# 4. ParentDocumentRetriever 구성
# 역할: 이 모든 구조를 연결하는 핵심 브레인입니다. 사용자가 질문을 던지면 
# (1) 자식 조각으로 정밀하게 검색한 뒤, 
# (2) 그 자식 조각이 속한 부모 문서를 찾아내어 LLM에게 전달합니다.
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=parent_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# ---------------------------------------------------------
# Pydantic 모델 정의
# ---------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str
    k: Optional[int] = 3

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]

# ---------------------------------------------------------
# API 엔드포인트
# ---------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Parent-Child RAG API Server is running!"}


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
        
        # 파일 형식별 로더 분기 및 인코딩 예외 처리
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        elif suffix in [".txt", ".md"]:
            try:
                # 1차 시도: utf-8 인코딩
                print("1차 시도: utf-8 인코딩")
                loader = TextLoader(tmp_path, encoding="utf-8")
                docs = loader.load()
            except Exception:
                # 2차 시도: 윈도우 기본 인코딩 (cp949 / euc-kr)
                print("2차 시도: 윈도우 기본 인코딩 (cp949 / euc-kr)")
                loader = TextLoader(tmp_path, encoding="cp949")
                docs = loader.load()
        else:
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식입니다: {suffix}")
        
        for doc in docs:
            doc.metadata["source"] = file.filename
            if category:
                doc.metadata["category"] = category
        
        # ParentDocumentRetriever를 통한 문서 추가
        retriever.add_documents(docs)
        os.unlink(tmp_path)
        
        return {
            "status": "success",
            "message": f"'{file.filename}' 파일이 Parent-Child 방식으로 성공적으로 색인되었습니다.",
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    # 검색 개수 설정 (retriever.search_kwargs = {"k": request.k}, 기본값 3)
    # 역할: 사용자의 질문과 가장 유사한 자식 조각을 몇 개나 찾아올 것인지 결정합니다. 
    # k=3이면 상위 3개의 자식 조각(과 그에 매핑되는 부모 문서들)을 가져와 참고 문서로 활용합니다.
    """
    작은 조각으로 정밀하게 검색한 후, 풍부한 문맥을 가진 부모 조각을 LLM에 전달합니다.
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="질문 내용이 비어 있습니다.")
        
        retriever.search_kwargs = {"k": request.k}
        
        template = """다음의 참고 문서를 바탕으로 질문에 정확하게 답변해 주세요.
만약 참고 문서에서 답을 찾을 수 없다면 "참고 문서에서 답을 찾을 수 없습니다."라고 답변해 주세요.

[참고 문서]
{context}

[질문]
{question}

[답변]:"""

        # 프롬프트 템플릿 (ChatPromptTemplate)
        # 역할: LLM에게 "참고 문서만을 바탕으로 답변할 것", "답을 찾을 수 없다면 모른다고 답할 것"이라는 
        # 엄격한 규칙(Grounding)을 부여하여 환각(Hallucination) 현상을 억제합니다.
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        retrieved_docs = retriever.invoke(request.question)
        sources = list(set([doc.metadata.get("source", "unknown") for doc in retrieved_docs]))
        answer = rag_chain.invoke(request.question)
        
        return AnswerResponse(
            question=request.question,
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))