
from dotenv import load_dotenv
from topic_guard import topic_guard

load_dotenv()

questions = [
    "What is the history of Jagannath Temple?",
    "Explain the Rath Yatra festival.",
    "What are the three main deities?",
    "How does Python programming work?",
    "Who won the latest cricket World Cup?"
]

for question in questions:
    try:
        topic_guard.validate(question)
        print("ALLOWED:", question)
    except Exception:
        print("BLOCKED:", question)
