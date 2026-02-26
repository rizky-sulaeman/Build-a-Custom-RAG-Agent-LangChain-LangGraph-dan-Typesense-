from retriever import TypesenseRetriever, simplify_hits


def pretty_print_hits(label, hits):
    print(f"\n=== {label} ===")
    if not hits:
        print("  (no results)")
        return

    for i, h in enumerate(hits, start=1):
        meta = h.get("metadata") or {}
        source = meta.get("source", "-")
        score = h.get("score")
        print(f"[{i}] id={h.get('id')}  source={source}  score={score}")
        print(f"content: {h.get('content')}")
        if meta:
            print(f"metadata: {meta}")


if __name__ == "__main__":
    retriever = TypesenseRetriever(collection_name="chunks", k=5)

    query = "Siloam"

    hits_text = simplify_hits(retriever.search(query, mode="text"))
    hits_vector = simplify_hits(retriever.search(query, mode="vector"))
    hits_hybrid = simplify_hits(retriever.search(query, mode="hybrid"))

    pretty_print_hits("TEXT", hits_text)
    pretty_print_hits("VECTOR", hits_vector)
    pretty_print_hits("HYBRID", hits_hybrid)