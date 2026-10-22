import streamlit as st
import requests



# ********* Streamlit 챗봇 UI



# FastAPI 서버 기본 URL
API_URL = "http://127.0.0.1:8000"

# 페이지 설정
st.set_page_config(
    page_title="RAG 기반 AI 대화 도우미",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG 기반 문서 대화 챗봇")

# ---------------------------------------------------------
# 사이드바: 파일 업로드 및 검색 옵션 설정
# ---------------------------------------------------------
with st.sidebar:
    st.header("📄 문서 업로드 & 설정")
    
    # 1. 파일 업로드 Form
    uploaded_file = st.file_uploader("문서 선택 (.txt, .pdf, .md)", type=["txt", "pdf", "md"])
    upload_tag = st.text_input("태그 입력 (선택)", value="default_tag")
    
    if st.button("문서 색인하기", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("문서를 분석하고 ChromaDB에 저장하는 중..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"tag": upload_tag}
                
                try:
                    res = requests.post(f"{API_URL}/upload", files=files, data=data)
                    if res.status_code == 200:
                        st.success(f"업로드 성공: {res.json().get('message')}")
                    else:
                        st.error(f"업로드 실패: {res.json().get('detail')}")
                except Exception as e:
                    st.error(f"서버 연결 오류: {e}")
        else:
            st.warning("업로드할 파일을 선택해 주세요.")

    st.divider()
    
    # 2. 질문 검색 필터 옵션
    st.header("🔍 검색 필터")
    search_tag = st.text_input("검색할 태그 (비워두면 전역 검색)", value="")
    top_k = st.slider("참고할 청크 개수 (k)", min_value=1, max_value=10, value=3)

# ---------------------------------------------------------
# 메인 화면: 챗봇 인터페이스
# ---------------------------------------------------------
# 세션 상태 초기화 (대화 기록 저장용)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 문서를 업로드하고 궁금한 점을 질문해 주세요."}
    ]

# 기존 대화 기록 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요..."):
    # 1. 사용자 메시지 화면에 표시 및 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. FastAPI 서버로 질문 전달 및 답변 수신
    with st.chat_message("assistant"):
        with st.spinner("하이브리드 검색 및 답변 생성 중..."):
            payload = {
                "question": prompt,
                "k": top_k
            }
            if search_tag.strip():
                payload["tag"] = search_tag.strip()
                
            try:
                response = requests.post(f"{API_URL}/ask", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "답변을 가져올 수 없습니다.")
                    sources = data.get("sources", [])
                    
                    # 출처 정보 추가 표시
                    full_response = answer
                    if sources:
                        full_response += f"\n\n---\n📌 **참고 문서:** {', '.join(sources)}"
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    error_msg = f"⚠️ 오류 발생: {response.json().get('detail', '알 수 없는 오류')}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"⚠️ 서버 연결 실패: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})