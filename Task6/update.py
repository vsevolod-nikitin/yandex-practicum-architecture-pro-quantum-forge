import json
import logging
from pathlib import Path
from datetime import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path("Task6/docs")
DATA_DIR = Path("Task3")
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "metadata.json"

MODEL_NAME = "BAAI/bge-m3"

CHUNK_SIZE = 750
CHUNK_OVERLAP = 150

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

def get_metadata() -> list:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return []

def save_metadata(meta: dict):
    META_PATH.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

def is_file_exist(path: Path, metadata: list) -> bool:
    for chunk in metadata:
        if chunk["source"] == str(path):
            return True

def chunk_document(file_path: Path, text: str, text_splitter):
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
    metadata = get_metadata()

    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()

    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
    else:
        index = faiss.IndexFlatIP(dim)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    new_vectors = []

    for file in DOCS_DIR.rglob("*"):
        if file.suffix.lower() not in {".txt"}:
            continue

        if is_file_exist(path=file, metadata=metadata):
            continue

        text = file.read_text(encoding="utf-8")

        chunks = chunk_document(file, text, text_splitter)

        texts = [f"passage: {c['text']}" for c in chunks]

        embeddings = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

        for _, emb in enumerate(embeddings):
            new_vectors.append(emb)

        for c in chunks:
            chunk_meta = {
                "id": c["id"],
                "source": c["metadata"]["source"],
                "chunk_index": c["metadata"]["chunk_index"],
                "total_chunks": c["metadata"]["total_chunks"],
                "text": c["text"],
            }
            metadata.append(chunk_meta)

    if new_vectors:
        index.add(np.array(new_vectors))
        logging.info(f"Added {len(new_vectors)} new chunks to index")
    else:
        logging.info("Added 0 new chunks to index")

    faiss.write_index(index, str(INDEX_PATH))
    save_metadata(metadata)

    index_size = index.ntotal
    index_file_size = INDEX_PATH.stat().st_size / (1024 * 1024)

    logging.info(
        f"Index stats: vectors={index_size}, file_size={index_file_size:.2f} MB"
    )

if __name__ == "__main__":
    start_time = datetime.now()
    logging.info(f"Start time: {str(start_time)}")
    try:
        main()
    except Exception:
        logging.exception("Index update failed")
    finally:
        end_time = datetime.now()
        logging.info(f"End time: {str(end_time)}")
        logging.info(f"Total time: {(end_time - start_time).seconds} seconds")