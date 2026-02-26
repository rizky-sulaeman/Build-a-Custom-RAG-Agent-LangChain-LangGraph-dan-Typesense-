from retriever import TypesenseRetriever, simplify_hits

retriever = TypesenseRetriever(collection_name="chunks", k=5)

query = "siloam"

# text search
hits_text = simplify_hits(retriever.search(query, mode="text"))

# vector search
hits_vector = simplify_hits(retriever.search(query, mode="vector"))

# hybrid search (recommended)
hits_hybrid = simplify_hits(retriever.search(query, mode="hybrid"))

print("TEXT:", hits_text)
print("VECTOR:", hits_vector)
print("HYBRID:", hits_hybrid)