"""
================================================================================
사업보고서 RAG 챗봇 (LlamaIndex + Gemini API + Supabase pgvector)
================================================================================
🔄 v2 변경 내역 (한글 PDF 추출 안정화)
   - PDF 리더를 SimpleDirectoryReader(pypdf 기반) → PyMuPDFReader로 교체
   - PDF 추출 직후 한글 자모 깨짐 자동 감지 + 경고 표시
   - requirements.txt에서 pymupdf 버전 고정 (1.24.10)
   - 회사명·메타데이터 NFC 정규화 추가

📚 이 코드가 하는 일을 한 줄로 요약하면:
   "PDF 사업보고서를 AI가 읽게 하고, 자연어로 질문하면 출처와 함께 답변하는 챗봇"

🛠 작동 원리 (RAG = Retrieval-Augmented Generation):
   1) 사업보고서 PDF를 작은 조각(청크)으로 자릅니다.
   2) 각 청크를 Gemini가 "숫자 벡터"로 변환합니다 (= 임베딩).
   3) 이 벡터들을 Supabase 벡터 DB에 저장합니다.
   4) 사용자가 질문하면, 질문도 벡터로 변환해 가장 비슷한 청크를 찾습니다.
   5) 찾은 청크를 Gemini에게 "참고하라"고 주면서 답변을 생성합니다.

📦 사용 도구:
   - Streamlit: 웹 챗봇 UI를 만드는 도구
   - LlamaIndex: RAG 파이프라인을 쉽게 만들어주는 라이브러리
   - PyMuPDF: PDF에서 텍스트를 추출하는 엔진 (한글에 강건)
   - Gemini API: LLM(답변 생성) + 임베딩(텍스트→벡터)
   - Supabase + pgvector: 벡터를 영구 저장하는 클라우드 DB
================================================================================
"""

# --------------------------------------------------------------------------
# [Section 0] 라이브러리 불러오기 (import)
# --------------------------------------------------------------------------
import streamlit as st              # 챗봇 웹페이지를 만드는 도구
import pandas as pd                  # 표(테이블) 데이터를 다루는 도구
import tempfile                      # 임시 폴더를 만드는 도구
import os                            # 파일 경로를 다루는 도구
import unicodedata                   # 한글 정규화 (NFC/NFD 통일용)

from supabase import create_client, Client  # Supabase(DB) 연결

# --- LlamaIndex 관련 도구들 (RAG의 핵심) ---
from llama_index.core import (
    VectorStoreIndex,                # 벡터 인덱스 = "임베딩으로 만든 검색 가능한 데이터 묶음"
    StorageContext,                  # 어디에 저장할지 알려주는 설정
    Settings,                        # LlamaIndex의 전역 설정 (LLM, 임베딩 모델 지정)
)

# 🔄 PDF 리더 교체: SimpleDirectoryReader(pypdf 기반) → PyMuPDFReader
# PyMuPDF는 MuPDF 엔진을 사용하여 한글 PDF 처리가 훨씬 안정적이에요.
# pypdf는 마이너 버전마다 한글 처리 로직이 바뀌어 같은 PDF에서도
# 결과가 달라지는 경우가 있는데, PyMuPDF는 그런 변동이 거의 없습니다.
from llama_index.readers.file import PyMuPDFReader

# --- Gemini API와 LlamaIndex를 연결하는 어댑터 ---
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig

# --- Supabase pgvector와 LlamaIndex를 연결하는 어댑터 ---
from llama_index.vector_stores.supabase import SupabaseVectorStore


# --------------------------------------------------------------------------
# [Section 1] 페이지 기본 설정
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="사업보고서 RAG 챗봇",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------------------------------
# [Section 2] 비밀 키(Secrets) 불러오기
# --------------------------------------------------------------------------
REQUIRED_KEYS = [
    "GEMINI_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "SUPABASE_DB_CONNECTION",
]

missing_keys = [k for k in REQUIRED_KEYS if k not in st.secrets]

if missing_keys:
    st.error(
        f"⚠️ Streamlit Secrets에 다음 키가 등록되지 않았습니다: {', '.join(missing_keys)}\n\n"
        "📋 등록 방법:\n"
        "1. Streamlit Cloud → 앱 우측 ⋮ 메뉴 → Settings → Secrets\n"
        "2. 아래 4개 키를 TOML 형식으로 입력:\n\n"
        '   GEMINI_API_KEY = "본인의 제미나이 키"\n'
        '   SUPABASE_URL = "https://xxxxx.supabase.co"\n'
        '   SUPABASE_KEY = "본인의 수파베이스 키"\n'
        '   SUPABASE_DB_CONNECTION = "postgresql://postgres.xxx:비밀번호@aws-0-xx.pooler.supabase.com:6543/postgres"\n\n'
        "3. Save 후 앱 재시작"
    )
    st.stop()

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
SUPABASE_DB_CONNECTION = st.secrets["SUPABASE_DB_CONNECTION"]


# --------------------------------------------------------------------------
# [Section 3] Supabase 및 LlamaIndex 초기화
# --------------------------------------------------------------------------
@st.cache_resource
def init_supabase() -> Client:
    """Supabase API 클라이언트를 만듭니다."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_resource
def init_llama_index():
    """
    LlamaIndex가 어떤 LLM과 임베딩 모델을 쓸지 전역 설정합니다.

    🧠 LLM: Gemini 2.5 Flash (무료 티어, 빠른 답변)
    🔢 임베딩: gemini-embedding-001 (3072 → 768차원으로 축소)
    📝 청크: 500자 단위, 50자 겹침
    """
    Settings.llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.1,  # 사업보고서는 정확성 우선
    )
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",
        api_key=GEMINI_API_KEY,
        embedding_config=EmbedContentConfig(
            output_dimensionality=768  # Matryoshka로 축소
        ),
    )
    Settings.chunk_size = 500
    Settings.chunk_overlap = 50


@st.cache_resource
def get_vector_store(company_name: str):
    """Supabase pgvector에 연결된 LlamaIndex 벡터 스토어를 반환합니다."""
    return SupabaseVectorStore(
        postgres_connection_string=SUPABASE_DB_CONNECTION,
        collection_name=company_name.replace(" ", "_").lower(),
        dimension=768,
    )


# --------------------------------------------------------------------------
# 🆕 [Section 3.5] PDF 추출 품질 검사 함수
# --------------------------------------------------------------------------
def check_korean_extraction_quality(documents):
    """
    PDF에서 추출된 텍스트가 한글 자모로 깨졌는지 페이지별로 검사합니다.

    📍 왜 필요한가?
       PDF 안의 폰트 매핑(ToUnicode CMap)이 손상돼 있으면,
       "안녕하세요" 같은 텍스트가 "ㅇㅏㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ"처럼
       자모로 분리되어 추출되는 경우가 있어요.

    📍 검사 방법:
       각 페이지의 텍스트에서:
       - 완성형 한글(가-힣) 개수
       - 한글 자모(ㄱ-ㅣ) 개수
       를 세어, 자모 비율이 비정상적으로 높으면 깨진 페이지로 판단합니다.

    반환값: 깨진 페이지 번호 리스트 (정상이면 빈 리스트)
    """
    broken_pages = []
    for doc in documents:
        text = doc.text.strip()
        # 거의 빈 페이지(차트만 있거나 표지 등)는 검사 생략
        if len(text) < 30:
            continue

        # 완성형 한글 글자 수 (가-힣)
        full_korean = sum(1 for c in text if '\uAC00' <= c <= '\uD7A3')
        # 한글 자모 글자 수 (ㄱ-ㅣ) — 정상 텍스트에는 거의 없어야 정상
        jamo = sum(1 for c in text if '\u3131' <= c <= '\u318E')

        # 자모가 완성형보다 많거나, 전체 텍스트의 5% 이상이면 깨진 페이지
        is_broken = (jamo > full_korean) or (jamo / len(text) > 0.05)
        if is_broken:
            page_label = doc.metadata.get("page_label", "?")
            broken_pages.append(page_label)

    return broken_pages


# 위의 함수들을 실제로 실행해서 사용 준비
supabase = init_supabase()
init_llama_index()


# --------------------------------------------------------------------------
# [Section 4] 화면 UI 구성
# --------------------------------------------------------------------------
st.title("📊 사업보고서 RAG 챗봇")
st.info(
    "💡 안내: PDF 사업보고서를 업로드하면, "
    "AI가 내용을 학습하고 자연어로 질문에 답해드립니다.\n\n"
    "📌 권장: 사업보고서 1개만 업로드해서 시작하세요. "
    "여러 개 올리면 검색 시 결과가 섞일 수 있어요."
)


# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_company" not in st.session_state:
    st.session_state.current_company = None
if "index" not in st.session_state:
    st.session_state.index = None


# --- 화면을 3개의 탭으로 나누기 ---
tab1, tab2, tab3 = st.tabs(["📤 업로드", "💬 챗봇", "📜 채팅 기록"])


# ==========================================================================
# [탭 1] 사업보고서 업로드 (PDF 인덱싱)
# ==========================================================================
with tab1:
    st.subheader("사업보고서 인덱싱")

    company_name = st.text_input(
        "🏢 회사명 입력",
        placeholder="예: 삼성전자, 카카오, 네이버",
        help="이 보고서가 어느 회사의 것인지 표시하기 위한 라벨입니다.",
    )

    uploaded_file = st.file_uploader(
        "📄 PDF 파일 선택",
        type=["pdf"],
        help="DART(전자공시시스템)에서 다운로드한 사업보고서 PDF를 업로드하세요.",
    )

    if uploaded_file is not None and company_name:
        if st.button("🚀 PDF 인덱싱 시작", type="primary"):
            with st.spinner("PDF 읽고 임베딩하는 중... (약 1-3분 소요)"):
                try:
                    # -----------------------------------------------------
                    # 단계 1: PDF를 임시 폴더에 저장
                    # -----------------------------------------------------
                    # 🔄 변경: 파일명을 영문 고정으로 — OS별 한글 파일명 인코딩 차이 회피
                    with tempfile.TemporaryDirectory() as temp_dir:
                        file_path = os.path.join(temp_dir, "document.pdf")
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # -----------------------------------------------------
                        # 단계 2: 🔄 PDF 읽기 (PyMuPDFReader 사용)
                        # -----------------------------------------------------
                        # PyMuPDFReader는 pypdf보다 한글 PDF 처리가 안정적입니다.
                        # 페이지 단위로 Document 객체 리스트를 반환해요.
                        reader = PyMuPDFReader()
                        documents = reader.load(file_path=file_path)

                        # PyMuPDFReader가 자동 설정하지 않는 메타데이터 보정
                        # (LlamaIndex의 다른 부분에서 page_label, file_name을 기대하므로)
                        original_filename = unicodedata.normalize(
                            "NFC", uploaded_file.name
                        )
                        for i, doc in enumerate(documents):
                            if "page_label" not in doc.metadata:
                                doc.metadata["page_label"] = str(i + 1)
                            doc.metadata["file_name"] = original_filename

                        # -----------------------------------------------------
                        # 🆕 단계 2.5: 한글 추출 품질 검사
                        # -----------------------------------------------------
                        # PyMuPDF가 처리하지 못한 페이지가 있는지 사전에 확인합니다.
                        # 자모로 깨진 텍스트가 그대로 임베딩되면 검색 품질이 크게 떨어져요.
                        broken_pages = check_korean_extraction_quality(documents)

                        if broken_pages:
                            pages_preview = ", ".join(broken_pages[:10])
                            if len(broken_pages) > 10:
                                pages_preview += f" ... (총 {len(broken_pages)}페이지)"
                            st.error(
                                f"⚠️ 다음 페이지에서 한글이 자모로 깨져 추출되었습니다:\n\n"
                                f"**페이지 {pages_preview}**\n\n"
                                "이 PDF는 폰트 매핑 정보가 손상된 페이지를 포함합니다. "
                                "다음을 시도해 보세요:\n"
                                "1. PDF를 다른 출처에서 다시 다운로드 (DART에서 Ctrl+P → PDF로 저장 권장)\n"
                                "2. 또는 OCR 처리된 PDF로 변환 후 재시도\n\n"
                                "인덱싱을 중단합니다. (깨진 데이터로 검색하면 정확하지 않은 답변이 생성됩니다)"
                            )
                            st.stop()

                        # -----------------------------------------------------
                        # 단계 3: 각 문서에 메타데이터(회사명) 추가
                        # -----------------------------------------------------
                        # 🔄 변경: 회사명을 NFC로 정규화 (Mac NFD 입력도 통일)
                        safe_company = unicodedata.normalize(
                            "NFC", company_name
                        ).strip()
                        for doc in documents:
                            doc.metadata["company"] = safe_company
                            # 모든 문자열 메타데이터도 NFC로 통일
                            for key, value in list(doc.metadata.items()):
                                if isinstance(value, str):
                                    doc.metadata[key] = unicodedata.normalize(
                                        "NFC", value
                                    )

                        # -----------------------------------------------------
                        # 단계 4: 벡터 스토어 + 인덱스 생성 (핵심!)
                        # -----------------------------------------------------
                        # 한 줄에서 자동으로 일어나는 일:
                        # 1) documents를 500자 청크로 자르기
                        # 2) 각 청크를 Gemini로 768차원 벡터로 임베딩
                        # 3) Supabase pgvector에 저장
                        vector_store = get_vector_store(safe_company)
                        storage_context = StorageContext.from_defaults(
                            vector_store=vector_store
                        )
                        index = VectorStoreIndex.from_documents(
                            documents,
                            storage_context=storage_context,
                            show_progress=True,
                        )

                        st.session_state.index = index
                        st.session_state.current_company = safe_company

                    # 성공 메시지
                    st.success(
                        f"✅ '{safe_company}' PDF 인덱싱 완료! "
                        f"({len(documents)} 페이지 처리)"
                    )
                    st.info("💬 챗봇 탭으로 이동해서 질문해보세요.")

                except Exception as e:
                    st.error(f"오류 발생: {e}")


# ==========================================================================
# [탭 2] 챗봇 (RAG로 질문 답변)
# ==========================================================================
with tab2:
    st.subheader("💬 사업보고서에 질문하기")

    if st.session_state.current_company:
        st.caption(f"📁 분석 대상: **{st.session_state.current_company}**")
    else:
        st.warning("⚠ 먼저 '업로드' 탭에서 사업보고서를 인덱싱해주세요.")

    # 이전 대화 화면에 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 참고 출처"):
                    for src in msg["sources"]:
                        st.caption(src)

    # 사용자 새 질문 입력
    if prompt := st.chat_input("질문을 입력하세요 (예: 작년 매출은?)"):
        if not st.session_state.index:
            st.error("먼저 사업보고서를 업로드해주세요.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    try:
                        # RAG 쿼리 실행 (검색 + 답변 생성)
                        query_engine = st.session_state.index.as_query_engine(
                            similarity_top_k=5,
                        )
                        response = query_engine.query(prompt)

                        answer = str(response)
                        st.markdown(answer)

                        # 출처 추출
                        sources = []
                        for node in response.source_nodes:
                            page = node.metadata.get("page_label", "?")
                            sources.append(
                                f"페이지 {page}: {node.text[:100]}..."
                            )

                        if sources:
                            with st.expander("📄 참고 출처"):
                                for src in sources:
                                    st.caption(src)

                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )

                        # Supabase chat_history 테이블에 대화 저장
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
                            st.toast(f"DB 저장 실패: {db_e}")

                    except Exception as e:
                        st.error(f"답변 생성 오류: {e}")


# ==========================================================================
# [탭 3] 채팅 기록 (Supabase에 저장된 과거 대화 보기)
# ==========================================================================
with tab3:
    st.subheader("📜 채팅 기록")

    try:
        response = (
            supabase.table("chat_history")
            .select("*")
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )

        if response.data:
            df = pd.DataFrame(response.data)
            df = df[["created_at", "company_name", "question", "answer"]]
            df.columns = ["시간", "회사", "질문", "답변"]

            st.dataframe(df, use_container_width=True, hide_index=True)

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
