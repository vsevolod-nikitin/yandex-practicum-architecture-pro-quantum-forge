from datetime import datetime
from pathlib import Path
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from sentence_transformers import SentenceTransformer

KNOWLEDGE_BASE_DIR = Path("Task2/knowledge_base")

CHUNK_SIZE = 750
CHUNK_OVERLAP = 150

OUTPUT_DIR = Path("Task3")
INDEX_PATH = OUTPUT_DIR / "faiss.index"
META_PATH = OUTPUT_DIR / "metadata.json"

MODEL_NAME = "BAAI/bge-m3"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

def get_documents():
    for file_path in KNOWLEDGE_BASE_DIR.rglob("*"):
        if file_path.suffix.lower() not in {".txt"}:
            continue

        text = file_path.read_text(encoding="utf-8")
        yield file_path, text


def chunk_document(file_path: Path, text: str):
    chunks = text_splitter.split_text(text)

    result = []
    for idx, chunk in enumerate(chunks):
        result.append(
            {
                "id": f"{file_path.stem}_{idx}",
                "text": chunk,
                "metadata": {
                    "source": str(file_path),
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                },
            }
        )
    return result

def main():
    total_chunks = []

    for file_path, text in get_documents():
        chunks = chunk_document(file_path, text)
        total_chunks.extend(chunks)

    texts = [f"passage: {c['text']}" for c in total_chunks]

    model = SentenceTransformer(MODEL_NAME)

    start_gen = datetime.now()
    embeddings = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    )
    finish_gen = datetime.now()

    embedding_dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    metadata = [
        {
            "id": c["id"],
            "source": c["metadata"]["source"],
            "chunk_index": c["metadata"]["chunk_index"],
            "total_chunks": c["metadata"]["total_chunks"],
            "text": c["text"],
        }
        for c in total_chunks
    ]

    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Total chunks: {len(total_chunks)}")
    print(f"Generation time: {(finish_gen - start_gen).seconds}")

if __name__ == "__main__":
    main()