import typesense, os

client = typesense.Client({
    "nodes": [{"host": "localhost", "port": "8108", "protocol": "http"}],
    "api_key": os.getenv("TYPESENSE_API_KEY"),
    "connection_timeout_seconds": 10,
})

# # list semua collection + jumlah dokumen
collections = client.collections.retrieve()
for c in collections:
    print(c["name"], c.get("num_documents"))

# print("Sample dokumen dari collection 'chunks':")
docs = client.collections["chunks"].documents.search({
    "q": "*",
    "query_by": "content",
    "per_page": 1,
})
print(docs)