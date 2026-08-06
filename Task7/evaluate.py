import json
import os
from datetime import datetime
from openai import OpenAI

from engine import RAGEngine

OPENROUTER_API_KEY = ""  # Необходимо указать ключ API OpenRouter

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

JUDGE_MODEL = "openai/gpt-oss-20b:free"

JUDGE_PROMPT = """
Вы нейтральный оценщик RAG-системы.

Вам будут даны:
- вопрос пользователя
- ожидаемый ответ
- фактический ответ, сгенерированный системой

Ваша задача:
1. Решить, корректен ли ответ.
2. Решить, правильно ли система себя повела (ответила или отказалась отвечать, если это уместно).
3. Обнаружить галлюцинации или неподтверждённые утверждения.

Возвращайте только "true", если фактический ответ похож на ожидаемый, иначе возвращайте "false"
"""

def judge_answer(question, expected, actual_answer):
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {
            "role": "user",
            "content": f"""
Вопрос:
{question}

Ожидаемый ответ:
{expected}

Фактический ответ:
{actual_answer}
""",
        },
    ]

    response = client.chat.completions.create(
        model=JUDGE_MODEL, messages=messages, temperature=0
    )

    if not response.choices:
        return None

    content = response.choices[0].message.content.strip()

    if "true" in content:
        return True
    elif "false" in content:
        return False
    else:
        return None


def main():
    rag = RAGEngine()
    with open("golden_questions.json", "r", encoding="utf-8") as f:
        golden = json.load(f)

    for qid, q in golden.items():
        question = q["question"]
        expected = q["expected_answer"]
        actual_answer = rag.answer(query=q["question"])

        evaluation = judge_answer(question, expected, actual_answer)

        log_entry = {
            "id": qid,
            "timestamp": str(datetime.now()),
            "success": evaluation,
            "question": question,
            "actual_answer": actual_answer,
            "expected_answer": expected,
        }

        with open("logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()