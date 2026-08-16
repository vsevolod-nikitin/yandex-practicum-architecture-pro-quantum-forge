import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from prompts import (
    SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES
)

MODEL_NAME = "BAAI/bge-m3"
LIMIT = 5

OPENROUTER_API_KEY = ""  # Необходимо указать ключ API OpenRouter

MALICIOUS_PATTERNS = ["ignore all instructions", "password", "root"]

def should_filter_chunk(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in MALICIOUS_PATTERNS)

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

class RAGEngine:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index("../Task3/faiss.index")

        with open("../Task3/metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def retrieve(self, query: str):
        emb = self.model.encode(f"query: {query}", normalize_embeddings=True)

        _, indices = self.index.search(np.array([emb]), LIMIT)

        chunks = []
        for idx in indices[0]:
            chunks.append(self.metadata[idx])

        return chunks

    def generate(self, query: str, chunks: list):
        if not chunks:
            return "Я не знаю."

        context = "\n\n".join(
            f"[{c['text']}" for c in chunks if not should_filter_chunk(c["text"])
        )

        examples = "\n\n".join(
            f"Q: {ex['question']}\nA: {ex['answer']}" for ex in FEW_SHOT_EXAMPLES
        )

        prompt = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {"role": "user", "content": f"Примеры:\n {examples}"},
            {"role": "user", "content": f"Контекст:\n {context}"},
            {"role": "user", "content": f"Вопрос:\n {query}"},
        ]

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b:free",
            messages=prompt,
        )

        if not response.choices:
            return "Я не знаю."

        return response.choices[0].message.content.strip()

    def answer(self, query: str):
        chunks = self.retrieve(query)
        return self.generate(query, chunks)