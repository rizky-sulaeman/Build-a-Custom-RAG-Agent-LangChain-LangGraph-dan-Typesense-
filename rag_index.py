import json  # Untuk serialisasi/deserialisasi data
import os  # Untuk akses environment variable
from typing import Iterable, Mapping, Any, List  # Tipe data untuk type hinting

import ollama as lama  # Library untuk koneksi ke OLLAMA (embeddings)
import typesense  # Library untuk koneksi ke Typesense (vector DB)



# Inisialisasi client Typesense, ambil konfigurasi dari environment variable
TYPESENSE_CLIENT = typesense.Client(
    {
        "nodes": [
            {
                "host": os.getenv("TYPESENSE_HOST", "localhost"),  # Default ke localhost
                "port": os.getenv("TYPESENSE_PORT", "8108"),  # Port default Typesense
                "protocol": os.getenv("TYPESENSE_PROTOCOL", "http"),
            }
        ],
        "api_key": os.getenv("TYPESENSE_API_KEY"),  # Wajib diisi
        "connection_timeout_seconds": 10,  # Timeout koneksi
    }
)



# Inisialisasi client OLLAMA, default ke endpoint cloudflare jika tidak ada env
OLLAMA_CLIENT = lama.Client(
    host=os.getenv(
        "OLLAMA_HOST",
        "https://boats-billing-kinds-detected.trycloudflare.com",
    )
)



# Nama model embedding, bisa diatur via env
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "hf.co/rizkysulaeman/Embedding-Gemma-300m-Healthcare:F16",
)



# Fungsi untuk generate embedding dari text pakai OLLAMA
def _embed(text: str) -> List[float]:
    rspn = OLLAMA_CLIENT.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text,
    )
    return rspn["embedding"]  # Ambil vektor embedding saja



# Pastikan collection Typesense sudah ada, kalau belum buat baru
def ensure_chunks_collection(name: str = "chunks") -> str:
    collections = [c["name"] for c in TYPESENSE_CLIENT.collections.retrieve()]
    if name in collections:
        return name  # Sudah ada, langsung pakai

    # Cari dimensi embedding secara dinamis dari sample
    sample_vec = _embed("sample")
    num_dim = len(sample_vec)

    # Definisi schema collection Typesense
    schema = {
        "name": name,
        "fields": [
            {"name": "id", "type": "string"},  # ID dokumen
            {"name": "content", "type": "string"},  # Isi dokumen
            {"name": "metadata", "type": "string", "optional": True},  # Metadata opsional
            {
                "name": "vector",
                "type": "float[]",
                "num_dim": num_dim,  # Dimensi embedding
            },
        ],
    }
    TYPESENSE_CLIENT.collections.create(schema)
    return name



# Normalisasi satu chunk: pastikan ada id, content, dan metadata
def _normalize_chunk(
    raw: Mapping[str, Any],
    id_fallback: int,
) -> Mapping[str, Any]:
    _id = str(raw.get("id") or id_fallback)  # Pakai id dari data, fallback ke urutan
    content = raw.get("content") or raw.get("text") or ""  # Ambil content, fallback ke text

    # Sisanya masuk ke metadata
    meta = {
        k: v
        for k, v in raw.items()
        if k not in {"id", "content", "text"}
    }

    return {
        "id": _id,
        "content": content,
        "metadata": json.dumps(meta, ensure_ascii=False),  # Metadata disimpan sebagai string JSON
    }



# Index banyak chunk ke Typesense, otomatis batch dan embed
def index_chunks(
    chunks: Iterable[Mapping[str, Any]],
    collection_name: str = "chunks",
    batch_size: int = 128,
) -> None:
    col = ensure_chunks_collection(collection_name)  # Pastikan collection siap

    docs: List[Mapping[str, Any]] = []
    for i, raw in enumerate(chunks, start=1):
        norm = _normalize_chunk(raw, id_fallback=i)  # Normalisasi data
        vec = _embed(norm["content"])  # Generate embedding
        norm["vector"] = vec  # Tambahkan vektor ke dokumen
        docs.append(norm)

    if not docs:
        return  # Tidak ada data, skip

    # Import ke Typesense per batch
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        TYPESENSE_CLIENT.collections[col].documents.import_(
            batch,
            {"action": "upsert"},  # Upsert: update jika sudah ada, insert jika baru
        )



# Index data dari file JSONL ke Typesense, hapus data lama dulu
def index_chunks_from_jsonl(
    path: str = "chunks.jsonl",
    collection_name: str = "chunks",
) -> None:

    chunks: List[Mapping[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip baris kosong
            chunks.append(json.loads(line))  # Parse JSON per baris

    # Hapus semua dokumen lama di collection sebelum index baru
    try:
        TYPESENSE_CLIENT.collections[collection_name].documents.delete({"filter_by": "id:!=null"})
        print(f"Nama Cchunk di document sebelumnya '{collection_name}' dihapus.")
    except Exception as e:
        print(f"Error ketika menghapus dokumen lama: {e}")
    index_chunks(chunks, collection_name=collection_name)



# Entry point: jalankan indexing dari file jika script dieksekusi langsung
if __name__ == "__main__":
    src = os.getenv("CHUNKS_JSONL", "chunks.jsonl")  # Path file sumber
    col = "chunks"  # Nama collection
    print(f"Indexing chunks from {src} into Typesense collection {col}...")
    index_chunks_from_jsonl(src, collection_name=col)
    print("Done.")

