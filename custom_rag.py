import os  # Untuk akses environment variable
from typing import Literal  # Untuk tipe literal pada return function

from langchain.tools import tool  # Dekorator untuk definisi tool
from langchain.chat_models import init_chat_model  # Inisialisasi model chat
from langchain_core.prompts import ChatPromptTemplate  # import ChatPromptTemplate
from langgraph.graph import MessagesState, StateGraph, START, END  # Untuk workflow graph
from langgraph.prebuilt import ToolNode, tools_condition  # Node dan kondisi tool
from pydantic import BaseModel, Field  # Untuk validasi dan schema output

from retriever import TypesenseRetriever, simplify_hits  # Import retriever custom


# Daftar provider model yang didukung (catatan saja)
"""Supported providers: anthropic, azure_al, azure_openal, bedrock, bedrock converse, cohere, deepseek, fireworks, google_anthropic_vertex, 
google genal, google vertexal, groq, huggingface, ibm, 
mistralal, nvidia, ollama, openal, perplexity, together, upstage, xal"""

# Nama model agent utama, bisa diubah lewat env
AGENT_MODEL_NAME = "gpt-oss:120b-cloud"  # Default, override via AGENT_MODEL

# Inisialisasi model untuk agent dan penilai relevansi
response_model = init_chat_model(AGENT_MODEL_NAME, temperature=0, model_provider="ollama")  # Model utama untuk generate response
model_penilai = init_chat_model(AGENT_MODEL_NAME, temperature=0, model_provider="ollama")  # Model untuk grading relevansi


""""
Typesense retriever tool (local database search):
- Menggunakan data lokal yang sudah di-index ke Typesense collection (default: "chunks")
- Data lokal sudah di-index ke Typesense collection (default: "chunks")
- Typesense dijalankan dengan vektor (vector search / hybrid) dan pencarian teks (text search)
- build chunking dan indexing sudah dilakukan sebelumnya, sehingga tool ini fokus untuk pencarian saja
"""

#1. Inisialisasi extends: `retrieve_chunks`
# Nama collection dan jumlah top_k hasil retrieval, bisa diubah via env
_DEFAULT_COLLECTION = os.getenv("CHUNKS_COLLECTION", "chunks")
_DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# Inisialisasi retriever Typesense
"""_ts_retriever adalah instance dari TypesenseRetriever yang sudah dikonfigurasi dengan nama collection dan jumlah top_k hasil retrieval.
- collection_name: Nama collection di Typesense tempat data chunk disimpan. Default "chunks", bisa diubah lewat environment variable CHUNKS_COLLECTION."
- k: Jumlah hasil teratas yang ingin diambil dari pencarian. Default 5, bisa diubah lewat environment variable RAG_TOP_K. Parameter ini menentukan berapa banyak hasil
 yang akan dikembalikan oleh retriever untuk setiap query yang diberikan.
"""
_ts_retriever = TypesenseRetriever(
    collection_name=_DEFAULT_COLLECTION,
    k=_DEFAULT_TOP_K,
)

#2. Tool: `retrieve_chunks` (pencarian ke Typesense)
# Tool untuk retrieval chunk dari Typesense

"""
retrieve_chunks adalah tool yang digunakan untuk mencari dan mengembalikan potongan dokumen (chunks) dari Typesense berdasarkan query yang diberikan.
- query: String pertanyaan atau kata kunci yang ingin dicari dalam koleksi Typesense
- Fungsi ini menggunakan retriever yang sudah diinisialisasi (_ts_retriever) untuk melakukan pencarian dengan mode "hybrid", yang menggabungkan pencarian teks dan vektor.
- Hasil pencarian disederhanakan dengan fungsi simplify_hits, kemudian konten dari setiap chunk yang ditemukan digabungkan menjadi satu string panjang yang akan dikembalikan sebagai output.
- Output berupa string yang berisi konten dari chunk yang relevan, dipisahkan dengan garis "---" antar chunk. Jika tidak ada hasil yang ditemukan, akan mengembalikan string kosong.
"""
@tool
def retrieve_chunks(query: str) -> str:
    """Cari dan kembalikan potongan dokumen lokal dari Typesense."""
    result = _ts_retriever.search(query, mode="hybrid")  # Cari dengan mode hybrid
    hits = simplify_hits(result)  # Sederhanakan hasil
    if not hits:
        return ""  # Kalau tidak ada hasil, return kosong

    # Gabungkan konten chunk jadi satu context panjang
    parts = []
    for h in hits:
        meta = h.get("metadata") or {}
        meta_str = ""
        if meta:
            meta_str = f"\n[metadata]: {meta}"  # Tambahkan metadata jika ada
        parts.append(
            f"[chunk_id={h.get('id')} score={h.get('score')}]\n{h.get('content','')}{meta_str}"
        )

    return "\n\n---\n\n".join(parts)  # Pisahkan antar chunk


# Alias tool untuk dipakai di agent
retriever_tool = retrieve_chunks

# 3. Node: Agent decide (generate_query_or_respond)
#    - Memutuskan: langsung jawab atau panggil tool `retrieve_chunks`

# Node agent: memutuskan apakah perlu retrieval atau langsung jawab
def generate_query_or_respond(state: MessagesState):
    """Panggil model untuk memutuskan: perlu retrieval atau langsung jawab.

    Jika perlu retrieval, model akan mengeluarkan tool_call ke `retrieve_chunks`.
    """
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])  # Bind tool dan invoke
    return {"messages": [response]}  # Kembalikan response

# 4. Relevance Check (grade_documents)

# Prompt untuk relevansi context
"""
G - Goal (Tujuan): Menentukan apa tujuan akhir dari permintaan. Apa yang ingin dicapai?
R - Request (Permintaan): Instruksi spesifik atau perintah yang harus dikerjakan oleh AI.
A - Action (Tindakan): Langkah-langkah atau peran (role) yang harus dilakukan AI. Misalnya, "Bertindaklah sebagai relevance of retrieved context to a user questio".
D - Details (Detail/Konteks): Informasi latar belakang, batasan, target audiens, atau informasi tambahan lainnya yang diperlukan.
E - Example (Contoh): Memberikan contoh format atau gaya jawaban yang diinginkan (zero-shot prompting). 
"""
"""
contoh dibawah ini adalah penerapan dari teknik prompting zero-shot prompting dengan menggunakan format GRADE (Goal, Request, Action, Details, Example) dan ChatPromptTemplate
untuk memberikan instruksi yang jelas kepada model dalam menilai relevansi konteks yang diambil dengan pertanyaan pengguna.
Karena kita menggunakan `with_structured_output`, kita HAPUS instruksi format JSON di dalam prompt karena model sudah akan mengembalikan output terstruktur otomatis.
"""
GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a grader assessing the relevance of a retrieved context to a user question.\n\n"
     "Goal (Tujuan): Determine the relevance of the retrieved context to the user's question.\n"
     "Request (Permintaan): Evaluate whether the context contains keywords or semantic meaning that can answer the question.\n"
     "Action (Tindakan): Act as a strict and objective document grader.\n"
     "Details (Detail/Konteks): You will be given a 'Retrieved context' and a 'User question'. A document is relevant if any part of it provides information directly related to the question.\n"
     "Example (Contoh): Provide your output strictly in JSON object format matching the requested schema. Set 'jawaban' to 'yes' if relevant, or 'no' if not."
    ),
    ("user", "Retrieved context:\n\n{context}\n\nUser question:\n{question}")
])


# Schema output 
"""
class dibawah ini adalah definisi schema output menggunakan Pydantic BaseModel untuk hasil grading relevansi.
- jawaban: Field yang menyatakan relevansi dalam bentuk string, dengan deskripsi bahwa nilainya harus "yes" jika relevan atau "no" jika tidak relevan. 
- Berkat `with_structured_output`, output respons akan secara konsisten bertipe `GradeDocuments` walau tanpa format instruction eksplisit pada prompt.
"""
class GradeDocuments(BaseModel):
    jawaban: str = Field(
        description="Relevance: 'yes' if relevant, or 'no' if not relevant"
    )

# Node relevansi: menentukan langkah selanjutnya
def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Menentukan apakah context hasil retrieval relevan dengan pertanyaan."""
    #ini adalah asumsi dari saya bahwa state["messages"] memiliki struktur di mana pesan pertama (messages[0]) 
    # adalah pertanyaan pengguna, dan pesan terakhir (messages[-1]) adalah hasil retrieval dari tool `retrieve_chunks`.
    # messages[0] = user question,
    # messages[-1] = ToolMessage dari `retrieve_chunks`
    question = state["messages"][0].content  # Ambil pertanyaan user
    context = state["messages"][-1].content  # Ambil context hasil retrieval

    # Gabung prompt dengan model via struktur `.with_structured_output(PydanticModel)`
    structured_llm = model_penilai.with_structured_output(GradeDocuments)
    grade_chain = GRADE_PROMPT | structured_llm
    
    response = grade_chain.invoke({"question": question, "context": context})  # Invoke chain
    nilai = (response.jawaban or "").strip().lower()  # Ambil hasil normalisasi jawaban

    if nilai == "yes":
        return "generate_answer"  # Kalau relevan, lanjut generate answer
    return "rewrite_question"  # Kalau tidak, rewrite pertanyaan



# 5. Rewrite Question (kalau retrieval pertama kali tidak relevan)
# Prompt untuk rewrite pertanyaan user
"""
Section ini adalah few-shot prompting untuk mendefinisikan prompt yang digunakan untuk merewrite pertanyaan pengguna agar lebih jelas dan relevan untuk pencarian di Typesense.
- Prompt ini memberikan instruksi kepada model untuk melihat pertanyaan asli dari pengguna, memahami konteksnya, dan kemudian merewrite pertanyaan tersebut agar lebih jelas dan lebih mungkin untuk mendapatkan hasil yang relevan dari indeks lokal Typesense.
- Dengan merewrite pertanyaan, diharapkan model dapat menangkap maksud sebenarnya dari pertanyaan pengguna, terutama jika pertanyaan awalnya ambigu atau kurang spesifik, sehingga meningkatkan kemungkinan mendapatkan hasil yang relevan dari pencarian di Typesense.
"""

REWRITE_PROMPT = (
    "You are a healthcare search assistant backed by a local knowledge base "
    "containing hospital information, doctors, and FAQs.\n"
    "Your task is to rewrite user questions so they are clearer, more specific, "
    "and optimized for retrieving relevant chunks from a local Typesense index.\n\n"

    "Examples:\n"

    "Original question: Where is Siloam Hospital located?\n"
    "Rewritten question: What are the locations and addresses of Siloam Hospitals in Indonesia?\n\n"

    "Original question: What inpatient rooms are available?\n"
    "Rewritten question: What types of inpatient rooms are offered by Siloam Hospitals?\n\n"

    "Original question: Is the blood supply safe in the hospital?\n"
    "Rewritten question: What safety standards are used for blood supply at Siloam Hospitals?\n\n"

    "Original question: What facilities does the hospital have?\n"
    "Rewritten question: What medical facilities and centers of excellence are available at Siloam Hospitals?\n\n"

    "Original question: Psychologist doctor in Yogyakarta?\n"
    "Rewritten question: Which psychologists are available at Siloam Hospitals Yogyakarta?\n\n"

    "Now rewrite the following user question:\n"
    "Original question:\n"
    "{question}\n"
    "Rewritten question:"
)


# Node rewrite pertanyaan: supaya retrieval lebih relevan
def rewrite_question(state: MessagesState):
    """Rewrite pertanyaan user supaya retrieval berikutnya lebih relevan."""
    messages = state["messages"]
    question = messages[0].content  # Ambil pertanyaan user
    prompt = REWRITE_PROMPT.format(question=question)  # Format prompt
    response = response_model.invoke([{"role": "user", "content": prompt}])  # Invoke model

    # Kembalikan sebagai HumanMessage baru, agar node berikutnya treat ini
    # seperti pertanyaan user yang sudah diperbaiki.
    from langchain_core.messages import HumanMessage
    return {"messages": [HumanMessage(content=response.content)]}  # Kembalikan pesan baru

# 6. Generate Final Answer
# Prompt untuk generate jawaban final
"""
Section ini adalah teknik Generated Knowledge prompting untuk mendefinisikan prompt yang digunakan dalam proses generate jawaban final.
- Prompt ini memberikan instruksi kepada model untuk pertama-tama menghasilkan pengetahuan latar belakang yang berguna berdasarkan konteks yang diambil, dan kemudian menggunakan pengetahuan tersebut untuk menghasilkan jawaban akhir.
- Dengan pendekatan ini, model tidak hanya mengandalkan konteks yang diambil, tetapi juga dapat menginferensi atau mengekstrak fakta penting yang mungkin tidak secara eksplisit disebutkan dalam konteks, sehingga dapat memberikan jawaban yang lebih informatif dan relevan terhadap pertanyaan pengguna.
"""
GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks over a local healthcare knowledge base.\n"
    "Before answering the question, first generate useful background knowledge based on the "
    "retrieved context. Then use that generated knowledge to produce the final answer.\n\n"

    "Instructions:\n"
    "Step 1 — Knowledge Generation:\n"
    "Extract or infer key facts, entities, and relationships from the provided context that may "
    "help answer the question. If the context is empty or insufficient, state that clearly.\n\n"

    "Step 2 — Final Answer:\n"
    "Use the generated knowledge together with the context to answer the users question.\n"
    "If the answer cannot be determined from the available data, say that you don't know "
    "based on the provided information.\n"
    "Keep the final answer concise (maximum 3 sentences).\n"

    "Question:\n{question}\n"
    "Context:\n{context}\n"

    "Output Format:\n"
    "Generated Knowledge:\n"
    "Final Answer:\n"
)



# Node generate jawaban final
def generate_answer(state: MessagesState):
    """Generate jawaban final menggunakan konteks yang sudah lolos relevance check."""
    question = state["messages"][0].content  # Ambil pertanyaan user
    context = state["messages"][-1].content  # Ambil context hasil retrieval
    prompt = GENERATE_PROMPT.format(question=question, context=context)  # Format prompt
    response = response_model.invoke([{"role": "user", "content": prompt}])  # Invoke model
    return {"messages": [response]}  # Kembalikan response



# 7. proses (Agent + RAG Flow)
"langgraph-hybrid-rag-tutorial.avif"

# proses graph LangGraph (Agent + RAG Flow)
def build_graph():
    workflow = StateGraph(MessagesState)  # Inisialisasi workflow

    # Tambahkan node ke graph
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

    # Start: dari user question ke agent decide
    workflow.add_edge(START, "generate_query_or_respond")

    # Agent decide: panggil tool atau langsung jawab
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",  # Kalau ada tool_calls → ke node `retrieve`
            END: END,  # Kalau tidak ada tool_calls → selesai (jawab final)
        },
    )

    # Setelah retrieval, relevance check → tentukan langkah berikutnya
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )

    # generate_answer → END
    workflow.add_edge("generate_answer", END)

    # rewrite_question → kembali ke agent decide
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    return workflow.compile()  # Compile graph



# Graph siap dipakai oleh aplikasi lain.
graph = build_graph()



# Contoh pemanggilan sederhana dari CLI
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    print("\nRunning Agentic RAG (LangGraph + Typesense)\n")
    while True:
        question = input("Pertanyaan kamu (atau ketik 'exit' untuk keluar): ")  # Input dari user
        if question.strip().lower() in ["exit", "quit"]:
            print("Keluar dari chat.")
            break
        if not question.strip():
            print("Pertanyaan tidak boleh kosong. Silakan masukkan pertanyaan.")
            continue
        state = {"messages": [HumanMessage(role="user", content=question)]}  # Bungkus pertanyaan
        for chunk in graph.stream(state):  # Stream hasil dari graph
            for node, update in chunk.items():
                msg = update["messages"][-1]
                content = getattr(msg, 'content', msg)
                # Handle output yang mengandung signature
                if isinstance(content, list) and content and isinstance(content[0], dict):
                    for item in content:
                        if 'extras' in item and 'signature' in item['extras']:
                            item = item.copy()
                            item['extras'] = {k: v for k, v in item['extras'].items() if k != 'signature'}
                    print(f"[{node}] -> {msg.type}: {content}\n")
                elif isinstance(content, dict) and 'extras' in content and 'signature' in content['extras']:
                    content = content.copy()
                    content['extras'] = {k: v for k, v in content['extras'].items() if k != 'signature'}
                    print(f"[{node}] -> {msg.type}: {content}\n")
                else:
                    print(f"[{node}] -> {msg.type}: {content}\n")

