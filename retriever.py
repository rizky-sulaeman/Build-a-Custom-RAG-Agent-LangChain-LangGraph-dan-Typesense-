
import json  # Untuk parsing dan serialisasi data dokumen
import os  # Untuk akses environment variable
from typing import List, Dict, Any, Literal 
import ollama as lama  # Client untuk model embedding
import typesense  # Client untuk Typesense search engine



# Mode pencarian retriever
SearchMode = Literal["text", "vector", "hybrid"]

# Inisialisasi client Typesense, konfigurasi dari environment variable
TYPESENSE_CLIENT = typesense.Client(
    {
        "nodes": [
            {
                "host": os.getenv("TYPESENSE_HOST", "localhost"),  # Default ke localhost
                "port": os.getenv("TYPESENSE_PORT", "8108"),  # Port default Typesense
                "protocol": os.getenv("TYPESENSE_PROTOCOL", "http"),
            }
        ],
        "api_key": os.getenv("TYPESENSE_API_KEY"),  # Wajib di-set agar bisa akses
        "connection_timeout_seconds": 10,  # Timeout koneksi
    }
)


# Client untuk OLLAMA, default host bisa diganti via env
OLLAMA_CLIENT = lama.Client(
    host=os.getenv(
        "OLLAMA_HOST",
        "https://boats-billing-kinds-detected.trycloudflare.com", # Endpoint OLLAMA default, bisa diganti dengan env
    )
)



# Nama model embedding yang dipakai
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "hf.co/rizkysulaeman/Embedding-Gemma-300m-Healthcare:F16",
)



# Fungsi untuk generate embedding dari teks pakai OLLAMA
def _embed(text: str) -> List[float]:
    rspn = OLLAMA_CLIENT.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text,
    )
    return rspn["embedding"]  # Ambil vektor embedding saja


# Retriever utama, bisa text, vector, atau hybrid search

class TypesenseRetriever:
    """
    Simple retriever di atas Typesense dengan 3 mode:
        - text
        - vector
        - hybrid
    """
    """Fungsi Dibawah ___init___ adalah untuk inisialisasi retriever dengan nama collection dan jumlah hasil yang diambil (k)."""
    def __init__(
        self,
        collection_name: str = "chunks", # Nama collection Typesense yang akan digunakan untuk pencarian
        k: int = 5, # Jumlah hasil yang ingin diambil dari pencarian, default 5. Bisa diubah saat panggil search() juga.
    ) -> None:
        self.collection_name = collection_name  # Nama koleksi Typesense
        self.k = k  # Default jumlah hasil yang diambil
    # "k" adalah parameter yang menentukan berapa banyak hasil yang ingin diambil dari pencarian. Misalnya,
    # jika k=5, maka retriever akan mengembalikan 5 hasil teratas yang paling relevan dengan query yang diberikan. Parameter ini bisa diatur saat 
    # inisialisasi retriever atau saat memanggil fungsi search() untuk fleksibilitas.
    def _search_text(self, query: str, k: int | None = None) -> Dict[str, Any]:
        """
        Search text adalah pencarian keyword di field 'content'.
        Biasanya lebih cepat, cocok untuk query yang sangat spesifik.
        """
        # Pencarian keyword biasa di field 'content'
        return TYPESENSE_CLIENT.collections[self.collection_name].documents.search(
            {
                "q": query,  # Query string dari user
                "query_by": "content",  # Field yang dicari
                "per_page": k or self.k,  # Jumlah hasil
            }
        )

    def _search_vector(self, query: str, k: int | None = None) -> Dict[str, Any]:
        """
        Search vector adalah pencarian berbasis kemiripan embedding (vector similarity).
        Cocok untuk query yang maknanya luas atau tidak harus exact match.
        """
        # Generate embedding dari query
        embedding = _embed(query)
        k = k or self.k
        # Format vector_query sesuai format Typesense
        vector_query = "vector:([{}], k:{})".format(
            ",".join(str(x) for x in embedding),
            k,
        )
        # multi_search untuk vector search
        body = {
            "searches": [
                {
                    "collection": self.collection_name,
                    "q": "*",  # Query wildcard, semua dokumen
                    "query_by": "content",
                    "vector_query": vector_query,
                    "per_page": k,
                }
            ]
        }
        multi = TYPESENSE_CLIENT.multi_search.perform(body)
        # multi_search returns {"results": [ ... ]}; ambil hasil pertama.
        return multi["results"][0]

    def _search_hybrid(
        self,
        query: str,
        k: int | None = None,
        # alpha: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Search hybrid adalah gabungan pencarian keyword dan vector (semantic).
        Biasanya hasil lebih relevan, karena menggabungkan dua pendekatan sekaligus. yaitu dengan 
        tetap mempertahankan query keyword untuk memastikan hasil yang benar-benar relevan, sambil juga memanfaatkan 
        pencarian berbasis embedding untuk menangkap makna yang lebih luas dari query.
        """
        # Hybrid: generate embedding + tetap pakai query keyword
        embedding = _embed(query)
        k = k or self.k
        vector_query = "vector:([{}], k:{})".format(
            ",".join(str(x) for x in embedding),
            k,
        )
        params: Dict[str, Any] = {
            "collection": self.collection_name,
            "q": query,  # Query keyword tetap dipakai
            "query_by": "content",
            "per_page": k,
            "vector_query": vector_query,  # Ditambah vector query
        }
        body = {"searches": [params]}
        multi = TYPESENSE_CLIENT.multi_search.perform(body)
        return multi["results"][0]

    def search(
        self,
        query: str,
        mode: SearchMode = "hybrid",
        k: int | None = None,
    ) -> Dict[str, Any]:
        # Pilih mode pencarian sesuai permintaan
        if mode == "text":
            return self._search_text(query, k=k)
        if mode == "vector":
            return self._search_vector(query, k=k)
        if mode == "hybrid":
            return self._search_hybrid(query, k=k)
        raise ValueError(f"Mode tidak dikenal: {mode}")  # Mode tidak dikenal


def simplify_hits(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Simplify hits adalah fungsi untuk mengambil bagian penting dari hasil pencarian Typesense.
    Typesense mengembalikan banyak informasi, tapi kita fokus ke id, content, metadata, dan skor relevansi.
    """
    out: List[Dict[str, Any]] = []
    for hit in result.get("hits", []):
        doc = hit.get("document", {})
        meta_str = doc.get("metadata")
        metadata = None
        # Metadata bisa string JSON atau None
        if isinstance(meta_str, str):
            try:
                metadata = json.loads(meta_str)
            except json.JSONDecodeError:
                metadata = meta_str  # Kalau gagal decode, pakai string as-is
        out.append(
            {
                "id": doc.get("id"),  # ID dokumen
                "content": doc.get("content"),  # Isi dokumen
                "metadata": metadata,  # Metadata sudah di-decode
                "score": hit.get("text_match") or hit.get("vector_distance"),  # Skor relevansi
            }
        )
    return out


