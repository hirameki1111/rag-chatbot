"""
================================================================================
사업보고서 RAG 챗봇 (LlamaIndex + Gemini API + Supabase pgvector)
================================================================================

📚 이 코드가 하는 일을 한 줄로 요약하면:
   "PDF 사업보고서를 AI가 읽게 하고, 자연어로 질문하면 출처와 함께 답변하는 챗봇"

🎓 비전공자 학습자에게:
   이 코드는 단순히 작동시키는 것보다 "왜 이렇게 짰는지" 이해하는 게 중요합니다.
   각 섹션마다 자세한 설명을 달아놓았으니 천천히 읽어보세요.

🛠 작동 원리 (RAG = Retrieval-Augmented Generation):
   1) 사업보고서 PDF를 작은 조각(청크)으로 자릅니다.
   2) 각 청크를 Gemini가 "숫자 벡터"로 변환합니다 (= 임베딩).
   3) 이 벡터들을 Supabase 벡터 DB에 저장합니다.
   4) 사용자가 질문하면, 질문도 벡터로 변환해 가장 비슷한 청크를 찾습니다.
   5) 찾은 청크를 Gemini에게 "참고하라"고 주면서 답변을 생성합니다.

📦 사용 도구:
   - Streamlit: 웹 챗봇 UI를 만드는 도구
   - LlamaIndex: RAG 파이프라인을 쉽게 만들어주는 라이브러리
   - Gemini API: LLM(답변 생성) + 임베딩(텍스트→벡터)
   - Supabase + pgvector: 벡터를 영구 저장하는 클라우드 DB

📝 사업보고서 업로드는 1개만 해도 충분합니다.
   여러 개 올리면 같은 DB에 섞여서 검색 결과가 혼합될 수 있어요.
   학습 목적이면 "삼성전자 1개"부터 시작하세요.
================================================================================
"""

# --------------------------------------------------------------------------
# [Section 0] 라이브러리 불러오기 (import)
# --------------------------------------------------------------------------
# Python 코드의 첫머리에는 항상 "어떤 도구를 쓸 건지" 선언합니다.
# 이것을 'import'라고 하고, 마치 책 한 권에서 필요한 도구를 꺼내쓰는 것과 같아요.

import streamlit as st              # 챗봇 웹페이지를 만드는 도구. st.title(), st.chat_input() 등 사용
import pandas as pd                  # 표(테이블) 데이터를 다루는 도구. 채팅 기록 보여줄 때 사용
import tempfile                      # 임시 폴더를 만드는 도구. 학생이 올린 PDF를 잠시 저장할 때 사용
import os                            # 파일 경로를 다루는 도구

from supabase import create_client, Client  # Supabase(우리의 DB)에 연결하는 도구

# --- LlamaIndex 관련 도구들 (RAG의 핵심) ---
from llama_index.core import (
    VectorStoreIndex,                # 벡터 인덱스 = "임베딩으로 만든 검색 가능한 데이터 묶음"
    SimpleDirectoryReader,           # 폴더 안의 파일들(PDF 등)을 읽는 도구
    StorageContext,                  # 어디에 저장할지 알려주는 설정
    Settings,                        # LlamaIndex의 전역 설정 (LLM, 임베딩 모델 지정)
)

# --- Gemini API와 LlamaIndex를 연결하는 어댑터 ---
from llama_index.llms.google_genai import GoogleGenAI            # Gemini를 LLM(답변 생성용)으로 쓰기 위한 어댑터
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding  # Gemini를 임베딩(텍스트→벡터)으로 쓰기 위한 어댑터
from google.genai.types import EmbedContentConfig                # 임베딩 모델 세부 설정(차원 수 등)

# --- Supabase pgvector와 LlamaIndex를 연결하는 어댑터 ---
from llama_index.vector_stores.supabase import SupabaseVectorStore


# --------------------------------------------------------------------------
# [Section 1] 페이지 기본 설정
# --------------------------------------------------------------------------
# Streamlit이 만든 웹페이지의 제목, 아이콘, 레이아웃을 설정합니다.
# 브라우저 탭에 보이는 정보를 정하는 곳이에요.

st.set_page_config(
    page_title="사업보고서 RAG 챗봇",  # 브라우저 탭 제목
    page_icon="📊",                      # 브라우저 탭 아이콘
    layout="wide",                       # 화면을 넓게 사용 (좌우 여백 줄임)
)


# --------------------------------------------------------------------------
# [Section 2] 비밀 키(Secrets) 불러오기
# --------------------------------------------------------------------------
# 우리 챗봇은 다음 4가지 외부 서비스를 사용합니다:
#   1) Gemini API (LLM + 임베딩)         → GEMINI_API_KEY 필요
#   2) Supabase API (대화 이력 저장)      → SUPABASE_URL, SUPABASE_KEY 필요
#   3) Supabase DB 직접 연결 (벡터 저장)  → SUPABASE_DB_CONNECTION 필요
#
# 이 키들은 비밀번호와 같아서 GitHub에 절대 직접 올리면 안 됩니다!
# 대신 Streamlit Cloud의 "Secrets"라는 안전한 보관함에 등록해야 해요.
#
# 만약 키가 등록되지 않은 채로 앱을 실행하면 에러가 나기 때문에,
# 친절한 안내 메시지를 보여주고 앱을 정지(st.stop)시킵니다.

REQUIRED_KEYS = [
    "GEMINI_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "SUPABASE_DB_CONNECTION",
]

# 등록되지 않은 키가 있는지 확인 (= 리스트 컴프리헨션)
missing_keys = [k for k in REQUIRED_KEYS if k not in st.secrets]

if missing_keys:
    # 학생이 잘 모를 수 있으니 자세한 안내를 보여줍니다
    st.error(
        f"⚠️ Streamlit Secrets에 다음 키가 등록되지 않았습니다: {', '.join(missing_keys)}\n\n"
        "📋 등록 방법:\n"
        "1. Streamlit Cloud → 앱 우측 ⋮ 메뉴 → Settings → Secrets\n"
        "2. 아래 4개 키를 TOML 형식으로 입력:\n\n"
        '   GEMINI_API_KEY = "본인의 제미나이 키"\n'
        '   SUPABASE_URL = "https://xxxxx.supabase.co"\n'
        '   SUPABASE_KEY = "본인의 수파베이스 키"\n'
        '   SUPABASE_DB_CONNECTION = "postgresql://postgres.xxx:비밀번호@aws-0-xx.pooler.supabase.com:6543/postgres"\n\n'
        "3. Save 후 앱 재시작\n\n"
        "💡 SUPABASE_DB_CONNECTION은 Supabase 대시보드 상단 'Connect' 버튼에서 가져옵니다."
    )
    st.stop()  # 키가 없으면 여기서 멈춤 (아래 코드 실행 X)

# 모든 키가 등록되어 있으면 변수에 담아 둡니다
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
SUPABASE_DB_CONNECTION = st.secrets["SUPABASE_DB_CONNECTION"]


# --------------------------------------------------------------------------
# [Section 3] Supabase 및 LlamaIndex 초기화
# --------------------------------------------------------------------------
# 외부 서비스에 연결하는 작업은 시간이 좀 걸립니다 (네트워크 통신).
# 매번 챗봇 페이지를 새로 그릴 때마다 다시 연결하면 너무 느리니까,
# @st.cache_resource 라는 "캐싱" 기능을 사용해 한 번만 연결하고 재사용합니다.
# (캐싱 = 결과를 잠시 저장해두고 다시 쓰는 기법)

@st.cache_resource
def init_supabase() -> Client:
    """
    Supabase API 클라이언트를 만듭니다.
    이걸로 chat_history 테이블에 대화 이력을 저장/조회할 수 있어요.
    """
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_resource
def init_llama_index():
    """
    LlamaIndex가 어떤 LLM과 임베딩 모델을 쓸지 전역 설정합니다.
    
    🧠 LLM이란?
       Large Language Model의 약자. "글을 이해하고 글을 쓰는 AI 모델".
       여기서는 Gemini 2.5 Flash를 씁니다 (무료 티어, 빠른 답변).
    
    🔢 임베딩이란?
       텍스트(글)를 "의미를 담은 숫자 묶음(벡터)"으로 바꾸는 작업.
       예: "휴학" → [0.23, -0.51, 0.89, ..., 0.42]  (768개 숫자)
       이 숫자들이 신기한 점은:
       - 의미가 비슷한 단어는 비슷한 숫자 패턴을 가짐
       - "휴학"과 "학업 중단"은 글자는 다르지만 벡터가 매우 비슷함
       - 그래서 글자가 달라도 의미로 검색할 수 있게 됨!
       
       gemini-embedding-001은 Google이 만든 최신 임베딩 모델로,
       기본은 3072차원이지만 768차원으로 축소해서 사용합니다.
       (Matryoshka Representation Learning이라는 기술 덕분에
        차원을 줄여도 성능 손실이 거의 없음)
    """
    # LLM: 답변을 생성하는 AI (gemini-2.5-flash — 빠르고 무료)
    Settings.llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.1,  # 0~1 사이 값. 낮을수록 일관된 답변, 높을수록 창의적 답변
                          # 사업보고서는 정확성이 중요하니 0.1로 낮게 설정
    )
    # 임베딩 모델: 텍스트를 768차원 벡터로 변환
    # ⚠️ 참고: text-embedding-004는 2026년 1월 deprecated되어 더 이상 사용 불가
    # 현재는 gemini-embedding-001을 사용 (기본 3072차원, output_dimensionality로 축소 가능)
    # 우리는 Supabase 테이블이 VECTOR(768)이므로 768차원으로 출력하도록 설정
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",
        api_key=GEMINI_API_KEY,
        embedding_config=EmbedContentConfig(
            output_dimensionality=768  # 3072 → 768로 축소 (Matryoshka)
        ),
    )
    # 청크(chunk) 크기 설정
    # 📝 청크란?
    #    "PDF의 텍스트를 작은 조각으로 자른 것"
    #    너무 크면 검색 정확도가 떨어지고, 너무 작으면 문맥이 사라집니다.
    #    500자 정도가 한국어/영어 모두에서 적당한 크기입니다.
    Settings.chunk_size = 500       # 청크당 글자 수
    Settings.chunk_overlap = 50     # 청크 간 겹치는 글자 수
                                    # 왜 겹치나? 문장이 청크 경계에서 잘려도
                                    # 옆 청크에 일부 포함되어 의미가 보존되게 하기 위함


@st.cache_resource
def get_vector_store(company_name: str):
    """
    Supabase pgvector에 연결된 LlamaIndex 벡터 스토어를 반환합니다.
    
    🗄 벡터 스토어란?
       임베딩된 청크들(=숫자 벡터)을 저장하는 데이터베이스.
       일반 DB는 "정확히 일치하는 글자"를 찾지만,
       벡터 DB는 "의미가 비슷한 벡터"를 찾을 수 있습니다.
    
    📂 collection_name이란?
       같은 DB 안에서 데이터를 묶는 "폴더" 같은 개념.
       회사명별로 다른 collection을 쓰면 나중에 회사별로 검색할 수도 있어요.
       (현재 코드는 collection만 나누고, 검색은 회사 무관하게 합니다)
    """
    return SupabaseVectorStore(
        postgres_connection_string=SUPABASE_DB_CONNECTION,
        # 회사명을 collection 이름으로 사용 (공백→_, 소문자로 변환)
        collection_name=company_name.replace(" ", "_").lower(),
        dimension=768,  # 임베딩 차원 수 (gemini-embedding-001을 768로 축소해서 사용)
    )


# 위의 함수들을 실제로 실행해서 사용 준비
supabase = init_supabase()
init_llama_index()


# --------------------------------------------------------------------------
# [Section 4] 화면 UI 구성 (Streamlit으로 챗봇 인터페이스 만들기)
# --------------------------------------------------------------------------
# Streamlit은 한 줄 한 줄이 곧 화면에 표시되는 요소가 됩니다.
# st.title()은 큰 제목, st.info()는 안내 메시지 박스를 만들어요.

st.title("📊 사업보고서 RAG 챗봇")
st.info(
    "💡 안내: PDF 사업보고서를 업로드하면, "
    "AI가 내용을 학습하고 자연어로 질문에 답해드립니다.\n\n"
    "📌 권장: 사업보고서 1개만 업로드해서 시작하세요. "
    "여러 개 올리면 검색 시 결과가 섞일 수 있어요."
)


# --- 세션 상태(session_state) 초기화 ---
# 🔄 세션 상태란?
#    Streamlit은 사용자가 버튼을 누를 때마다 코드가 처음부터 다시 실행됩니다.
#    그러면 변수들이 사라져버리는데, st.session_state에 저장하면
#    페이지가 새로 그려져도 값이 유지됩니다.
#    (마치 브라우저 탭의 "메모리"와 같은 역할)
if "messages" not in st.session_state:
    st.session_state.messages = []           # 챗봇 대화 내용 보관 (질문/답변 리스트)
if "current_company" not in st.session_state:
    st.session_state.current_company = None  # 현재 분석 중인 회사 이름
if "index" not in st.session_state:
    st.session_state.index = None            # 인덱싱된 벡터 인덱스


# --- 화면을 3개의 탭으로 나누기 ---
# 탭 = 가로로 나란히 있는 메뉴. 클릭하면 해당 화면이 보임.
tab1, tab2, tab3 = st.tabs(["📤 업로드", "💬 챗봇", "📜 채팅 기록"])


# ==========================================================================
# [탭 1] 사업보고서 업로드 (PDF 인덱싱)
# ==========================================================================
# 사용자가 PDF를 업로드하면 다음 과정을 거칩니다:
#
#   📄 PDF 파일
#      ↓ (LlamaIndex가 텍스트 추출)
#   📝 텍스트 덩어리들 (페이지별)
#      ↓ (자동 청킹 - 500자씩 자르기)
#   🧱 청크들 (예: 200페이지 → 약 1,500개 청크)
#      ↓ (Gemini 임베딩 - 각 청크를 768차원 벡터로 변환)
#   🔢 임베딩 벡터들 (1,500개의 벡터)
#      ↓ (Supabase pgvector에 저장)
#   💾 벡터 DB에 영구 저장 완료!
# ==========================================================================
with tab1:
    st.subheader("사업보고서 인덱싱")

    # 회사명 입력 — 검색 결과를 추적하기 위한 라벨
    company_name = st.text_input(
        "🏢 회사명 입력",
        placeholder="예: 삼성전자, 카카오, 네이버",
        help="이 보고서가 어느 회사의 것인지 표시하기 위한 라벨입니다.",
    )

    # PDF 파일 업로더 — 학생이 사업보고서 PDF를 선택하면 메모리에 임시 저장됨
    uploaded_file = st.file_uploader(
        "📄 PDF 파일 선택",
        type=["pdf"],  # PDF 파일만 허용
        help="DART(전자공시시스템)에서 다운로드한 사업보고서 PDF를 업로드하세요.",
    )

    # 파일과 회사명이 모두 입력되면 버튼 활성화
    if uploaded_file is not None and company_name:
        if st.button("🚀 PDF 인덱싱 시작", type="primary"):
            # st.spinner: 작업하는 동안 빙글빙글 로딩 아이콘 표시
            with st.spinner("PDF 읽고 임베딩하는 중... (약 1-3분 소요)"):
                try:
                    # -----------------------------------------------------
                    # 단계 1: PDF를 임시 폴더에 저장
                    # -----------------------------------------------------
                    # LlamaIndex의 SimpleDirectoryReader는 "폴더 단위로 읽기" 때문에,
                    # 업로드된 파일을 임시 폴더에 잠시 저장해줘야 합니다.
                    # tempfile.TemporaryDirectory()는 with 블록이 끝나면 자동으로 삭제됩니다.
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # 임시 폴더에 PDF 저장
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # -----------------------------------------------------
                        # 단계 2: PDF 읽기 (텍스트 추출 + 자동 청킹)
                        # -----------------------------------------------------
                        # SimpleDirectoryReader는 PDF를 페이지 단위로 읽고,
                        # 각 페이지의 텍스트를 Document 객체로 만들어줍니다.
                        # Section 3에서 설정한 chunk_size=500에 따라
                        # 나중에 자동으로 청크로 잘립니다.
                        documents = SimpleDirectoryReader(
                            input_dir=temp_dir
                        ).load_data()

                        # -----------------------------------------------------
                        # 단계 3: 각 문서에 메타데이터(회사명) 추가
                        # -----------------------------------------------------
                        # 메타데이터 = "데이터에 대한 데이터" (책의 표지 정보 같은 것)
                        # 나중에 검색 결과에서 "어느 회사 보고서에서 찾았는지" 알 수 있게 함
                        for doc in documents:
                            doc.metadata["company"] = company_name

                        # -----------------------------------------------------
                        # 단계 4: 벡터 스토어 + 인덱스 생성 (핵심!)
                        # -----------------------------------------------------
                        # 이 한 줄이 RAG의 핵심 마법입니다:
                        # 1) documents를 청크로 자르고
                        # 2) 각 청크를 Gemini로 임베딩하고 (768차원 벡터)
                        # 3) Supabase pgvector에 저장
                        # 위 세 작업이 from_documents() 한 번에 자동으로 일어나요!
                        vector_store = get_vector_store(company_name)
                        storage_context = StorageContext.from_defaults(
                            vector_store=vector_store
                        )
                        index = VectorStoreIndex.from_documents(
                            documents,
                            storage_context=storage_context,
                            show_progress=True,  # 진행 상황 콘솔에 표시
                        )

                        # 만들어진 인덱스를 세션 상태에 저장 → 챗봇 탭에서 사용
                        st.session_state.index = index
                        st.session_state.current_company = company_name

                    # 성공 메시지
                    st.success(
                        f"✅ '{company_name}' PDF 인덱싱 완료! "
                        f"({len(documents)} 페이지 처리)"
                    )
                    st.info("💬 챗봇 탭으로 이동해서 질문해보세요.")

                except Exception as e:
                    # 에러 발생 시 사용자에게 알려주기
                    st.error(f"오류 발생: {e}")


# ==========================================================================
# [탭 2] 챗봇 (RAG로 질문 답변)
# ==========================================================================
# 사용자가 질문을 입력하면 다음 과정을 거칩니다:
#
#   ❓ 사용자 질문 ("작년 매출은?")
#      ↓ (질문도 Gemini로 임베딩)
#   🔢 질문 벡터 (768차원)
#      ↓ (Supabase pgvector에서 유사도 검색)
#   📚 가장 비슷한 청크 5개 선택
#      ↓ (찾은 청크 + 질문을 Gemini에게 전달)
#   🤖 Gemini가 답변 생성 (검색된 청크에 근거)
#      ↓
#   💬 답변 + 출처 페이지 표시
# ==========================================================================
with tab2:
    st.subheader("💬 사업보고서에 질문하기")

    # 현재 분석 중인 회사 표시
    if st.session_state.current_company:
        st.caption(f"📁 분석 대상: **{st.session_state.current_company}**")
    else:
        st.warning("⚠ 먼저 '업로드' 탭에서 사업보고서를 인덱싱해주세요.")

    # --- 이전 대화 화면에 표시 ---
    # 세션 상태에 저장된 메시지를 모두 그리기
    # (사용자가 새 질문을 할 때마다 코드가 다시 실행되므로 매번 그려야 함)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):  # role: "user" 또는 "assistant"
            st.markdown(msg["content"])
            # 출처가 있으면 접을 수 있는 형태로 표시
            if msg.get("sources"):
                with st.expander("📄 참고 출처"):
                    for src in msg["sources"]:
                        st.caption(src)

    # --- 사용자 새 질문 입력 ---
    # st.chat_input은 화면 하단에 채팅 입력창을 만들어줍니다.
    # 사용자가 엔터를 누르면 입력값이 prompt 변수에 담겨요.
    if prompt := st.chat_input("질문을 입력하세요 (예: 작년 매출은?)"):
        # 인덱스가 없으면 (= PDF 업로드 안 했으면) 에러
        if not st.session_state.index:
            st.error("먼저 사업보고서를 업로드해주세요.")
        else:
            # 사용자 질문을 메시지 리스트에 추가 + 화면에 표시
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # AI 답변 생성
            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    try:
                        # -----------------------------------------------------
                        # RAG 쿼리 실행 (검색 + 답변 생성)
                        # -----------------------------------------------------
                        # as_query_engine()은 "질문 답변 엔진"을 만듭니다.
                        # similarity_top_k=5 → 가장 비슷한 청크 5개를 찾아서 LLM에게 줌
                        #   (너무 적으면 정보 부족, 너무 많으면 노이즈와 비용 증가)
                        query_engine = st.session_state.index.as_query_engine(
                            similarity_top_k=5,
                        )

                        # query()를 호출하면 LlamaIndex가 자동으로:
                        # 1) prompt를 임베딩 벡터로 변환
                        # 2) Supabase에서 가장 비슷한 청크 5개 검색
                        # 3) 찾은 청크들을 Gemini에게 "참고하라"고 전달
                        # 4) Gemini가 답변 생성
                        # 5) response.source_nodes에 어떤 청크를 참고했는지 정보 담김
                        response = query_engine.query(prompt)

                        # 답변 텍스트만 추출해서 화면에 표시
                        answer = str(response)
                        st.markdown(answer)

                        # -----------------------------------------------------
                        # 출처 추출 (어떤 청크를 참고했는지)
                        # -----------------------------------------------------
                        # response.source_nodes: 답변 생성에 사용된 청크들의 리스트
                        # 각 청크의 metadata에서 페이지 번호와 텍스트 일부를 가져옴
                        sources = []
                        for node in response.source_nodes:
                            page = node.metadata.get("page_label", "?")
                            # 청크 텍스트 앞 100자만 미리보기로 표시
                            sources.append(
                                f"페이지 {page}: {node.text[:100]}..."
                            )

                        # 출처를 접을 수 있는 박스로 표시
                        if sources:
                            with st.expander("📄 참고 출처"):
                                for src in sources:
                                    st.caption(src)

                        # AI 답변을 메시지 리스트에 저장 (다음 페이지 그릴 때 표시되도록)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )

                        # -----------------------------------------------------
                        # Supabase chat_history 테이블에 대화 저장 (영구 기록)
                        # -----------------------------------------------------
                        # 세션 상태는 페이지를 닫으면 사라지지만,
                        # Supabase에 저장하면 영구 보관되어 나중에도 볼 수 있어요.
                        try:
                            supabase.table("chat_history").insert(
                                {
                                    "question": prompt,
                                    "answer": answer,
                                    "sources": sources,
                                    "company_name": st.session_state.current_company,
                                }
                            ).execute()
                        except Exception as db_e:
                            # DB 저장 실패해도 답변은 보여주기 (사소한 오류로 처리)
                            st.toast(f"DB 저장 실패: {db_e}")

                    except Exception as e:
                        st.error(f"답변 생성 오류: {e}")


# ==========================================================================
# [탭 3] 채팅 기록 (Supabase에 저장된 과거 대화 보기)
# ==========================================================================
# Supabase의 chat_history 테이블에서 데이터를 가져와 표로 보여줍니다.
# CSV로 다운로드도 가능해서 나중에 분석 자료로 쓸 수 있어요.
# ==========================================================================
with tab3:
    st.subheader("📜 채팅 기록")

    try:
        # Supabase에서 채팅 기록 가져오기
        # .select("*"): 모든 컬럼 가져오기
        # .order("created_at", desc=True): 최신순 정렬
        # .limit(50): 최근 50개만 (너무 많으면 느려짐)
        response = (
            supabase.table("chat_history")
            .select("*")
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )

        if response.data:
            # 가져온 데이터를 표(DataFrame)로 변환
            df = pd.DataFrame(response.data)
            # 보기 좋게 필요한 컬럼만 선택하고 한글로 바꾸기
            df = df[["created_at", "company_name", "question", "answer"]]
            df.columns = ["시간", "회사", "질문", "답변"]

            # 표 화면에 표시
            st.dataframe(df, use_container_width=True, hide_index=True)

            # --- CSV 다운로드 버튼 ---
            # utf-8-sig 인코딩: 한글이 Excel에서 깨지지 않게 BOM 추가
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 CSV로 다운로드",
                csv,
                "chat_history.csv",
                "text/csv",
            )
        else:
            st.info("아직 저장된 대화가 없습니다. 챗봇 탭에서 질문해보세요!")

    except Exception as e:
        st.error(f"채팅 기록 불러오기 실패: {e}")
