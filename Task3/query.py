import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from build_index import INDEX_PATH, META_PATH, MODEL_NAME

LIMIT = 5

def main():
    query = input("Request: ")

    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(f"query: {query}", normalize_embeddings=True)

    index = faiss.read_index(str(INDEX_PATH))

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    scores, indices = index.search(np.array([query_embedding]), LIMIT)

    print("\nResults:\n")

    for score, idx in zip(scores[0], indices[0]):
        chunk = metadata[idx]
        print(f"Score: {score:.4f}")
        print(f"Source: {chunk['source']}")
        print(f"Text: {chunk['text'][:150]}...")
        print("=" * 70)

if __name__ == "__main__":
    main()