"""
사업보고서 RAG 챗봇
====================
PDF 사업보고서를 업로드하면 자연어로 질문할 수 있는 RAG 챗봇.

기술 스택
---------
- UI         : Streamlit
- RAG        : LlamaIndex
- Vector DB  : Supabase (pgvector)
- LLM        : Google Gemini 2.5 Flash
- Embedding  : Google gemini-embedding-001 (768차원)
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime
from typing import Any

import streamlit as st
from supabase import Client, create_client

# LlamaIndex 핵심 모듈 (전역 설정·인덱스·스토리지 컨텍스트)
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.supabase import SupabaseVectorStore


# =============================================================================
# 0. 페이지 기본 설정
#    - Streamlit 앱이 시작될 때 가장 먼저 1번만 호출되어야 한다.
# =============================================================================
st.set_page_config(
    page_title="사업보고서 RAG 챗봇",
    page_icon="📊",
    layout="wide",
)


# =============================================================================
# 1. 시크릿 로드
#    - .streamlit/secrets.toml 또는 Streamlit Cloud Secrets 패널에서 읽어온다.
#    - 키가 누락되면 즉시 사용자에게 안내 후 앱을 중단한다 (st.stop).
# =============================================================================
def load_secrets() -> dict[str, str]:
    """필수 시크릿을 한 번에 읽어 검증한다."""
    required_keys = [
        "GEMINI_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "SUPABASE_DB_CONNECTION",
    ]
    missing = [k for k in required_keys if k not in st.secrets]
    if missing:
        st.error(
            "❌ 다음 시크릿이 .streamlit/secrets.toml에 없습니다:\n\n"
            + "\n".join(f"- {k}" for k in missing)
        )
        st.stop()
    return {k: st.secrets[k] for k in required_keys}


SECRETS = load_secrets()


# =============================================================================
# 2. 캐싱된 초기화 함수
#    - @st.cache_resource: 무거운 자원(클라이언트·모델 등)을 1회만 만들고
#      이후 모든 사용자 세션이 재활용 → 비용 절약 & 속도 향상
# =============================================================================
@st.cache_resource(show_spinner=False)
def init_supabase() -> Client:
    """
    Supabase REST API 클라이언트 초기화.

    chat_history 테이블 CRUD에 사용된다.
    """
    return create_client(SECRETS["SUPABASE_URL"], SECRETS["SUPABASE_KEY"])


@st.cache_resource(show_spinner=False)
def init_llama_index() -> bool:
    """
    LlamaIndex의 전역 Settings에 LLM·임베딩·청크 설정을 등록.

    Settings는 싱글톤이므로 한 번만 설정하면 이후 모든 인덱스/쿼리에 적용된다.
    """
    # LLM: 답변 생성용 (temperature 낮춰서 일관된 답변)
    Settings.llm = Gemini(
        model="models/gemini-2.5-flash",
        api_key=SECRETS["GEMINI_API_KEY"],
        temperature=0.1,
    )

    # 임베딩: 문서를 벡터로 변환 (768차원으로 명시 → DB 컬럼과 일치 필요)
    Settings.embed_model = GeminiEmbedding(
        model_name="models/gemini-embedding-001",
        api_key=SECRETS["GEMINI_API_KEY"],
        output_dimensionality=768,
    )

    # 청크: 문서를 잘게 쪼개는 단위. 500자 + 50자 겹침
    Settings.chunk_size = 500
    Settings.chunk_overlap = 50
    return True


def _normalize_collection_name(company_name: str) -> str:
    """
    회사명을 Supabase collection_name 규칙에 맞게 정규화.

    - 소문자 변환
    - 공백·특수문자를 언더스코어(_)로 치환
    - 영문/숫자/언더스코어만 남김
    """
    name = company_name.strip().lower().replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    return name.strip("_") or "default"


@st.cache_resource(show_spinner=False)
def get_vector_store(company_name: str) -> SupabaseVectorStore:
    """
    회사별 SupabaseVectorStore 인스턴스를 반환.

    Parameters
    ----------
    company_name : str
        회사명. 내부에서 소문자·언더스코어 정규화하여 collection_name으로 사용.
    """
    return SupabaseVectorStore(
        postgres_connection_string=SECRETS["SUPABASE_DB_CONNECTION"],
        collection_name=_normalize_collection_name(company_name),
        dimension=768,
    )


# 앱 시작 시 LlamaIndex 전역 설정을 미리 적용
init_llama_index()
supabase: Client = init_supabase()


# =============================================================================
# 3. 핵심 도메인 함수
# =============================================================================
def build_index_from_pdf(pdf_file, company_name: str) -> int:
    """
    업로드된 PDF를 임베딩하여 Supabase pgvector에 저장.

    Returns
    -------
    int
        인덱싱된 청크(노드) 수
    """
    # 1) 업로드된 파일을 임시 디렉터리에 저장 (LlamaIndex가 경로 기반으로 읽음)
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, pdf_file.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.read())

        # 2) PDF 로딩 (페이지별로 Document 객체 생성)
        documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()

        # 3) 각 Document에 회사명 메타데이터 추가 (필터링·출처 표시용)
        for doc in documents:
            doc.metadata["company_name"] = company_name

        # 4) Supabase Vector Store 준비
        vector_store = get_vector_store(company_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 5) 인덱싱 (자동으로 청크 분할 → 임베딩 → DB 저장)
        index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            show_progress=False,
        )

        # 인덱싱된 노드(청크) 개수 추정
        return len(index.docstore.docs)


def ask_question(company_name: str, question: str) -> tuple[str, list[dict[str, Any]]]:
    """
    질의에 대한 답변과 출처 페이지 정보를 반환.

    Returns
    -------
    answer : str
    sources : list[dict]
        [{"page": 3, "score": 0.82, "snippet": "..."}, ...]
    """
    # 기존에 적재된 Vector Store에서 인덱스 로드
    vector_store = get_vector_store(company_name)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # similarity_top_k: 유사 청크 몇 개를 LLM에 함께 전달할지
    query_engine = index.as_query_engine(similarity_top_k=4)
    response = query_engine.query(question)

    # 출처 추출 (LlamaIndex는 source_nodes에 청크 메타데이터를 담아줌)
    sources: list[dict[str, Any]] = []
    for node in response.source_nodes:
        meta = node.node.metadata or {}
        # pypdf 기반 리더는 "page_label" 키로 페이지를 제공
        page = meta.get("page_label") or meta.get("page_number") or "?"
        sources.append(
            {
                "page": str(page),
                "score": round(float(node.score), 3) if node.score is not None else None,
                "snippet": node.node.get_content()[:120].replace("\n", " ") + "…",
            }
        )

    return str(response), sources


def save_chat_history(
    question: str, answer: str, sources: list[dict[str, Any]], company_name: str
) -> None:
    """대화 1건을 chat_history 테이블에 저장."""
    supabase.table("chat_history").insert(
        {
            "question": question,
            "answer": answer,
            "sources": sources,
            "company_name": company_name,
        }
    ).execute()


def fetch_chat_history(company_name: str | None = None, limit: int = 50) -> list[dict]:
    """대화 이력을 최신순으로 조회."""
    query = supabase.table("chat_history").select("*").order("created_at", desc=True).limit(limit)
    if company_name:
        query = query.eq("company_name", company_name)
    return query.execute().data or []


def fetch_company_list() -> list[str]:
    """저장된 회사명 목록을 중복 제거하여 반환."""
    rows = (
        supabase.table("chat_history")
        .select("company_name")
        .not_.is_("company_name", "null")
        .execute()
        .data
        or []
    )
    return sorted({r["company_name"] for r in rows if r.get("company_name")})


# =============================================================================
# 4. 화면 구성
# =============================================================================
st.title("📊 사업보고서 RAG 챗봇")
st.caption("PDF 사업보고서를 업로드하고 자연어로 질문하세요. 답변에는 출처 페이지가 함께 표시됩니다.")

# 세션 상태 초기화: 챗봇 탭에서 메시지 기록 유지
if "messages" not in st.session_state:
    st.session_state.messages = []

tab_upload, tab_chat, tab_history = st.tabs(["📤 업로드", "💬 챗봇", "📜 채팅 기록"])


# -----------------------------------------------------------------------------
# Tab 1. 업로드 - PDF 인덱싱
# -----------------------------------------------------------------------------
with tab_upload:
    st.subheader("PDF 사업보고서 업로드")

    col1, col2 = st.columns([2, 1])
    with col1:
        pdf_file = st.file_uploader("사업보고서 PDF 선택", type=["pdf"])
    with col2:
        company_name_input = st.text_input(
            "회사명",
            placeholder="예: 삼성전자",
            help="이 회사명으로 벡터 컬렉션이 생성/구분됩니다.",
        )

    if st.button("🚀 인덱싱 실행", type="primary", disabled=not (pdf_file and company_name_input)):
        try:
            with st.spinner("PDF를 분석하고 벡터 DB에 저장하는 중입니다... (수십 초~수 분 소요)"):
                node_count = build_index_from_pdf(pdf_file, company_name_input.strip())
            st.success(
                f"✅ **{company_name_input}** 인덱싱 완료! "
                f"총 **{node_count}개**의 청크가 저장되었습니다."
            )
            st.balloons()
        except Exception as e:  # noqa: BLE001
            st.error(f"❌ PDF 처리 중 오류가 발생했습니다: {e}")


# -----------------------------------------------------------------------------
# Tab 2. 챗봇 - 질의응답
# -----------------------------------------------------------------------------
with tab_chat:
    st.subheader("질의응답")

    try:
        companies = fetch_company_list()
    except Exception as e:  # noqa: BLE001
        st.error(f"❌ DB 연결 오류: {e}")
        companies = []

    # 업로드는 했지만 아직 대화 기록이 없는 회사도 고를 수 있도록 직접 입력 옵션 제공
    target_company = st.selectbox(
        "분석 대상 회사 선택",
        options=["(직접 입력)"] + companies,
        index=0,
    )
    if target_company == "(직접 입력)":
        target_company = st.text_input("회사명 입력", value="", placeholder="업로드 탭에서 사용한 회사명")

    # 기존 메시지 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 출처 보기"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"- **p.{s['page']}** "
                            f"(유사도 {s['score']}) — {s['snippet']}"
                        )

    user_q = st.chat_input("사업보고서에 대해 궁금한 점을 입력하세요")
    if user_q:
        if not target_company:
            st.warning("⚠️ 먼저 회사명을 선택하거나 입력해 주세요.")
        else:
            # 사용자 메시지 표시
            st.session_state.messages.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            # 답변 생성
            with st.chat_message("assistant"):
                with st.spinner("답변을 생성하는 중..."):
                    try:
                        answer, sources = ask_question(target_company, user_q)
                    except Exception as e:  # noqa: BLE001
                        answer, sources = f"❌ 답변 생성 중 오류: {e}", []

                st.markdown(answer)
                if sources:
                    with st.expander("📎 출처 보기"):
                        for s in sources:
                            st.markdown(
                                f"- **p.{s['page']}** "
                                f"(유사도 {s['score']}) — {s['snippet']}"
                            )

            # 세션 상태 + DB 저장
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            try:
                save_chat_history(user_q, answer, sources, target_company)
            except Exception as e:  # noqa: BLE001
                st.warning(f"⚠️ 대화 이력 저장 실패: {e}")


# -----------------------------------------------------------------------------
# Tab 3. 채팅 기록 - chat_history 조회
# -----------------------------------------------------------------------------
with tab_history:
    st.subheader("저장된 대화 이력")

    try:
        companies = fetch_company_list()
    except Exception as e:  # noqa: BLE001
        st.error(f"❌ DB 연결 오류: {e}")
        companies = []

    filter_company = st.selectbox(
        "회사 필터",
        options=["(전체)"] + companies,
        index=0,
    )

    try:
        rows = fetch_chat_history(
            company_name=None if filter_company == "(전체)" else filter_company,
            limit=100,
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"❌ DB 조회 오류: {e}")
        rows = []

    if not rows:
        st.info("아직 저장된 대화가 없습니다.")
    else:
        st.caption(f"총 **{len(rows)}**건 (최신순)")
        for row in rows:
            created = row.get("created_at", "")
            try:
                # ISO 문자열을 보기 좋게 포맷
                created = datetime.fromisoformat(created.replace("Z", "+00:00")).strftime(
                    "%Y-%m-%d %H:%M"
                )
            except Exception:  # noqa: BLE001
                pass

            title = f"🕒 {created} · 🏢 {row.get('company_name', '-')} · ❓ {row['question'][:40]}…"
            with st.expander(title):
                st.markdown(f"**Q.** {row['question']}")
                st.markdown(f"**A.** {row['answer']}")
                src = row.get("sources") or []
                if src:
                    st.markdown("**📎 출처**")
                    for s in src:
                        st.markdown(
                            f"- p.{s.get('page', '?')} "
                            f"(유사도 {s.get('score', '-')}) — {s.get('snippet', '')}"
                        )