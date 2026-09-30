from embeddings import EmbeddingModel


embedder = EmbeddingModel()

texts = [
    "Python functions allow code to be reused.",
    "Functions in Python can accept parameters and return values.",
    "A database stores and retrieves structured information.",
]

embeddings = embedder.embed(texts)

print("Embedding shape:")
print(embeddings.shape)

print("\nFirst embedding:")
print(embeddings[0])

print("\nNumber of dimensions:")
print(len(embeddings[0]))
