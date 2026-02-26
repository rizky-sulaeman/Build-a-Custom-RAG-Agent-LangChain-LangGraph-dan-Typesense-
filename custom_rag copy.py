import os
from typing import Literal

from langchain.tools import tool
from langchain_core.messages import convert_to_messages
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from retriever import TypesenseRetriever, simplify_hits


# -----------------------------------------------------------------------------
# LLM setup (agent + grader) – uses LangChain's init_chat_model API.
# You can override the model via the AGENT_MODEL env var.
# -----------------------------------------------------------------------------

"Supported providers: anthropic, azure_ai, azure_openai, bedrock, bedrock_converse, cohere, deepseek, fireworks, google_anthropic_vertex, google_genai, google_vertexai, groq, huggingface, ibm, mistralai, nvidia, ollama, openai, perplexity, together, upstage, xai"


# Use Google Gemini (genai) for tool-calling support
AGENT_MODEL_NAME = "gpt-oss:20b-cloud"  # Bisa diubah via env AGENT_MODEL

response_model = init_chat_model(AGENT_MODEL_NAME, temperature=0, model_provider="ollama")  # Pastikan API key sudah diset di env
grader_model = init_chat_model(AGENT_MODEL_NAME, temperature=0, model_provider="ollama")  # Pastikan API key sudah diset di env


# -----------------------------------------------------------------------------
# Typesense retriever tool (local data + FastEmbed-style semantic search)
#
# Assumsi:
# - Data lokal sudah di-index ke Typesense collection (default: "chunks")
#   via skrip lain di repo ini (mis. rag_index / build_chunks).
# - Typesense dijalankan dengan dukungan vektor (mis. FastEmbed / vector
#   field seperti yang dipakai di file lain).
# -----------------------------------------------------------------------------

_DEFAULT_COLLECTION = os.getenv("CHUNKS_COLLECTION", "chunks")
_DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

_ts_retriever = TypesenseRetriever(
    collection_name=_DEFAULT_COLLECTION,
    k=_DEFAULT_TOP_K,
)


@tool
def retrieve_chunks(query: str) -> str:
    """Cari dan kembalikan potongan dokumen lokal dari Typesense."""
    result = _ts_retriever.search(query, mode="hybrid")
    hits = simplify_hits(result)
    if not hits:
        return ""

    # Satukan konten sebagai context panjang.
    parts = []
    for h in hits:
        meta = h.get("metadata") or {}
        meta_str = ""
        if meta:
            meta_str = f"\n[metadata]: {meta}"
        parts.append(
            f"[chunk_id={h.get('id')} score={h.get('score')}]\n{h.get('content','')}{meta_str}"
        )

    return "\n\n---\n\n".join(parts)


retriever_tool = retrieve_chunks


# -----------------------------------------------------------------------------
# 3. Node: Agent decide (generate_query_or_respond)
#    - Memutuskan: langsung jawab atau panggil tool `retrieve_chunks`
# -----------------------------------------------------------------------------


def generate_query_or_respond(state: MessagesState):
    """Panggil model untuk memutuskan: perlu retrieval atau langsung jawab.

    Jika perlu retrieval, model akan mengeluarkan tool_call ke `retrieve_chunks`.
    """
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


# -----------------------------------------------------------------------------
# 4. Relevance Check (grade_documents)
# -----------------------------------------------------------------------------

GRADE_PROMPT = (
    "You are a grader assessing relevance of retrieved context to a user question.\n"
    "Here is the retrieved context:\n\n{context}\n\n"
    "Here is the user question:\n{question}\n\n"
    "If the context contains keyword(s) or semantic meaning related to the question, "
    "grade it as relevant.\n"
    "Return a binary score: 'yes' if relevant, 'no' if not relevant."
)


class GradeDocuments(BaseModel):
    """Binary relevance score for retrieved context."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Menentukan apakah context hasil retrieval relevan dengan pertanyaan."""
    # Asumsi: messages[0] = user question,
    # messages[-1] = ToolMessage dari `retrieve_chunks`
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    score = (response.binary_score or "").strip().lower()

    if score == "yes":
        return "generate_answer"
    return "rewrite_question"


# -----------------------------------------------------------------------------
# 5. Rewrite Question (kalau retrieval pertama kali tidak relevan)
# -----------------------------------------------------------------------------

REWRITE_PROMPT = (
    "Look at the user question below and infer the underlying intent.\n"
    "You are talking to a healthcare search assistant backed by a local knowledge base.\n"
    "Original question:\n"
    "-------\n"
    "{question}\n"
    "-------\n"
    "Rewrite this question so that it is clearer and more likely to retrieve relevant chunks "
    "from the local Typesense index.\n"
)


def rewrite_question(state: MessagesState):
    """Rewrite pertanyaan user supaya retrieval berikutnya lebih relevan."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])

    # Kembalikan sebagai HumanMessage baru, agar node berikutnya treat ini
    # seperti pertanyaan user yang sudah diperbaiki.
    from langchain_core.messages import HumanMessage

    return {"messages": [HumanMessage(content=response.content)]}


# -----------------------------------------------------------------------------
# 6. Generate Final Answer
# -----------------------------------------------------------------------------

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks over a local healthcare knowledge base.\n"
    "Use the retrieved context below to answer the question. If the context is empty or "
    "insufficient, say that you don't know based on the available data.\n"
    "Keep the answer concise (max 3 sentences).\n\n"
    "Question:\n{question}\n\n"
    "Context:\n{context}\n"
)


def generate_answer(state: MessagesState):
    """Generate jawaban final menggunakan konteks yang sudah lolos relevance check."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# -----------------------------------------------------------------------------
# 7. Rakitan Graph LangGraph (Agent + RAG Flow)
#
# Flow:
#   User Question
#       ↓
#   generate_query_or_respond  (Agent Decide: perlu retrieval atau tidak)
#       ↓ (tools_condition)
#       ├── tools → retrieve (Typesense tool) → grade_documents
#       │                               ├── generate_answer → END
#       │                               └── rewrite_question → generate_query_or_respond
#       └── END (kalau agent langsung jawab tanpa retrieval)
# -----------------------------------------------------------------------------


def build_graph():
    workflow = StateGraph(MessagesState)

    # Nodes
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

    # Start
    workflow.add_edge(START, "generate_query_or_respond")

    # Agent decide: panggil tool atau langsung jawab
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",  # kalau ada tool_calls → jalan ke node `retrieve`
            END: END,  # kalau tidak ada tool_calls → selesai (jawab final)
        },
    )

    # Setelah retrieval, lakukan relevance check → tentukan langkah berikutnya
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )

    # generate_answer → END
    workflow.add_edge("generate_answer", END)

    # rewrite_question → kembali ke agent decide
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    return workflow.compile()


# Graph siap dipakai oleh aplikasi lain.
graph = build_graph()


if __name__ == "__main__":
    # Contoh pemanggilan sederhana dari CLI:
    from langchain_core.messages import HumanMessage

    print("\n=== Running Agentic RAG (LangGraph + Typesense) ===\n")
    from langchain_core.messages import HumanMessage
    while True:
        question = input("Pertanyaan kamu (atau ketik 'exit' untuk keluar): ")
        if question.strip().lower() in ["exit", "quit"]:
            print("Keluar dari chat.")
            break
        if not question.strip():
            print("Pertanyaan tidak boleh kosong. Silakan masukkan pertanyaan.")
            continue
        print("[LOG] User question diterima.")
        state = {"messages": [HumanMessage(role="user", content=question)]}
        print("[LOG] Memproses dengan graph.invoke()...")
        result = graph.invoke(state)
        # Format hasil agar mudah dibaca dan terstruktur:
        print("[LOG] Ringkasan hasil graph.invoke():")
        messages = result.get("messages", [])
        for idx, msg in enumerate(messages):
            role = getattr(msg, "role", msg.__class__.__name__)
            content = getattr(msg, "content", str(msg))
            print(f"  [{idx}] {role}:")
            print(f"Konten: {content}")
            if msg.__class__.__name__ == "ToolMessage" and hasattr(msg, "content"):
                print("    [Dokumen RAG Referensi]")
                for chunk in content.split("\n---\n\n"):
                    lines = chunk.strip().split("\n")
                    chunk_id = lines[0] if lines else ""
                    preview = lines[1] if len(lines)>1 else ""
                    print(f"-{chunk_id}")
                    print(f" {preview}")
        final_response = result["messages"][-1].content
        print(f"\nJawaban: {final_response}\n")


