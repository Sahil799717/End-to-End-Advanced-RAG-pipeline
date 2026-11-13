from guardrails import Guard
from guardrails_ai.restricttotopic import RestrictToTopic

topic_guard = Guard().use(
    RestrictToTopic(
        valid_topics=[
            "Jagannath Temple, its history, architecture, deities, rituals, Rath Yatra, and religious traditions of Odisha"
        ],
        invalid_topics=[
            "Python programming and software development",
            "Cricket and other sports",
            "Politics",
            "Finance and business",
            "Unrelated general knowledge"
        ],
        device=-1,
        model="facebook/bart-large-mnli",
        disable_classifier=False,
        disable_llm=True,
        zero_shot_threshold=0.3,
        on_fail="exception"
    )
)