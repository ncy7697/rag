🚀 Advanced Local RAG Pipeline
Python 3.11+, FastAPI, ChromaDB, 그리고 로컬 LLM(Ollama)을 기반으로 구축된 고성능 로컬 고급 RAG(Advanced RAG) 백엔드 시스템입니다.
대용량 문서의 맥락 손실을 막기 위한 Parent-Child Chunking과 검색 정확도를 극대화하는 FlashRank Reranker가 적용되어 있습니다.

🛠️ Tech Stack
Language: Python 3.11+

Framework: FastAPI

RAG / Orchestration: LangChain (langchain-core, langchain-community, langchain-chroma, langchain-ollama)

Vector Store: ChromaDB (Local Persistent)

LLM & Embedding (Local): Ollama (llama3.2, nomic-embed-text)

Advanced RAG Features:

Parent-Child Chunking (부모 1500자 / 자식 400자)

FlashRank Reranker (Cross-Encoder 기반 2차 정밀 재순위화)

인코딩 자동 예외 처리 (utf-8 / cp949 이중 시도)

📂 Project Architecture
Plaintext
.
├── main_parent_child.py    # FastAPI 백엔드 메인 코드 (Parent-Child + Reranker)
├── chroma_db_parent_child/ # ChromaDB 벡터 스토어 영구 저장 디렉토리
└── README.md
⚙️ Installation & Setup
1. 가상환경 생성 및 활성화
Bash
python -m venv .venv313
# Windows 기준
.venv313\Scripts\activate
2. 필수 패키지 설치
Bash
pip install fastapi uvicorn pypdf python-multipart langchain langchain-core langchain-community langchain-chroma langchain-ollama flashrank
3. Ollama 모델 다운로드 및 실행
로컬에 Ollama가 설치되어 있어야 하며, 다음 모델들을 미리 받아둡니다.

Bash
ollama pull llama3.2
ollama pull nomic-embed-text
🚀 Running the Server
백엔드 서버를 실행합니다.

Bash
uvicorn main_parent_child:app --reload
서버 주소: [http://127.0.0.1:8000](http://127.0.0.1:8000)

API 문서(Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

🔌 API Endpoints
1. 문서 업로드 및 색인 (POST /upload)
설명: .txt, .md, .pdf 파일을 업로드받아 자동으로 Parent-Child 구조로 분할한 뒤 ChromaDB에 색인합니다. (윈도우 인코딩 자동 호환)

Parameter: file (Multipart File)

2. 질문 및 답변 생성 (POST /ask)
설명: 자식 조각으로 1차 정밀 검색 후, FlashRank Reranker로 재채점하여 최적의 부모 문서를 선별한 뒤 LLM 답변을 생성합니다.

Request Body Example:

JSON
{
  "question": "프로젝트 핵심 목표가 무엇인가요?",
  "k": 5
}
Response Example:

JSON
{
  "question": "프로젝트 핵심 목표가 무엇인가요?",
  "answer": "참고 문서에 따르면 프로젝트의 핵심 목표는...",
  "sources": ["1014.txt"]
}
💡 Key Architectural Highlights
Parent-Child Chunking

검색은 400자(Child) 단위로 아주 날카롭게 수행하고, LLM에게 전달할 때는 1500자(Parent) 단위의 풍부한 전후 맥락을 제공하여 정보 유실을 방지합니다.

FlashRank Reranker

1차로 검색된 후보 문서들을 가벼운 CPU 환경에서도 빠르게 동작하는 Cross-Encoder로 재채점하여, 관련 없는 노이즈 문서를 걸러내고 환각(Hallucination) 현상을 차단합니다.

InMemoryStore & 인코딩 방어 로직

부모 문서 저장 시 발생할 수 있는 직렬화 오류를 메모리 스토어(InMemoryStore)로 원천 방지하였으며, 텍스트 파일 로드 시 utf-8 실패 시 cp949로 자동 폴백되도록 안정성을 높였습니다.