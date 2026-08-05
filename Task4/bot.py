from time import sleep

from engine import RAGEngine

rag = RAGEngine()

def main():
    while True:
        query = input("Введите вопрос:")

        answer = rag.answer(query)        
        print(answer)


if __name__ == "__main__":
    main()