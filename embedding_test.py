import ollama as lama

client = lama.Client(host='https://boats-billing-kinds-detected.trycloudflare.com') 

text = "Halo, apa kabar?"

rspn = client.embeddings(
    model="hf.co/rizkysulaeman/Embedding-Gemma-300m-Healthcare:F16",
    prompt=text,
)

vec = rspn["embedding"]
print(len(vec))
print(rspn)             