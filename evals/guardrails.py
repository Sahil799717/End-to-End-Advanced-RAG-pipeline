
from dotenv import load_dotenv
from guardrails import Guard
from langchain_groq import ChatGroq

from guardrails_ai.detect_pii import DetectPII
from guardrails_ai.detect_jailbreak import DetectJailbreak
from guardrails_ai.restricttotopic import RestrictToTopic
from guardrails_ai.response_evaluator import ResponseEvaluator

load_dotenv()

GROQ_GUARD_MODEL = "groq/openai/gpt-oss-20b"

# Groq model for topic classification
guardrail_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=2,
    timeout=90,
)


def classify_topic(prompt):
    response = guardrail_llm.invoke(prompt)
    return str(response.content)


# Input validators
pii_validator = DetectPII(
    entities=["EMAIL_ADDRESS", "PHONE_NUMBER"],
    on_fail="fix",
)

topic_validator = RestrictToTopic(
    valid_topics=[
        "Jagannath Temple history and origins",
        "Jagannath Temple architecture",
        "Lord Jagannath and temple deities",
        "Temple rituals and worship practices",
        "Rath Yatra and temple festivals",
        "Temple traditions and pilgrimage",
    ],
    invalid_topics=[
        "Politics",
        "Sports",
        "Entertainment",
        "Programming",
        "Technology",
        "Finance",
        "Unrelated general knowledge",
    ],
    disable_classifier=True,
    disable_llm=False,
    llm_callable=classify_topic,
    on_fail="exception",
)

# Retrieved context validator
jailbreak_validator = DetectJailbreak(
    threshold=0.8,
    on_fail="exception",
)

# Generated response validator
response_validator = ResponseEvaluator(
    llm_callable=GROQ_GUARD_MODEL,
    on_fail="exception",
)


# Separate guards
input_guard = Guard().use(
    pii_validator,
    topic_validator,
)

context_guard = Guard().use(
    jailbreak_validator,
)

response_guard = Guard().use(
    response_validator,
)


def validate_input(query):
    result = input_guard.validate(query)
    return result.validated_output or query


def validate_context(context):
    context_guard.validate(context)
    return context


def validate_response(answer, query, context):
    validation_question = f"""
User question:
{query}

Retrieved context:
{context}

Does the answer correctly answer the user's question
and is it supported by the retrieved context?

Answer YES only if both conditions are satisfied.
Otherwise answer NO.
"""

    result = response_guard.validate(
        answer,
        metadata={
            "validation_question": validation_question,
        },
    )

    return result.validated_output or answer
