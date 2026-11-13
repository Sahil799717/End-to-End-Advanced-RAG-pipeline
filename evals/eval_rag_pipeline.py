
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)

import rag_backend

from config import (
    PDF_PATH,
    EMBEDDING_MODEL,
    RETRIEVAL_K,
    RERANK_TOP_N,
)

from evals.judge import GroqEvaluationLLM
from evals.harness import (
    load_goldens,
    summarize_by_metric,
    print_summary,
)


GOLDEN_PATH = "goldens/temples_goldens.json"
THRESHOLD = 0.7


def run(agent):
    goldens = load_goldens(GOLDEN_PATH)
    judge = GroqEvaluationLLM()
    test_cases = []

    for golden in goldens:
        result = rag_backend.ask_question(
            golden["query"],
            agent_instance=agent,
        )

        if not result["is_relevant"]:
            print(f"Skipped insufficient context: {golden['query']}")
            continue

        context = [
            doc.page_content
            for doc in result["source_documents"]
            if doc.page_content
        ]

        if not context or not result["answer"]:
            print(f"Skipped empty result: {golden['query']}")
            continue

        reference_context = golden["reference_context"]

        if isinstance(reference_context, str):
            reference_context = [reference_context]

        test_cases.append(
            LLMTestCase(
                input=golden["query"],
                actual_output=result["answer"],
                expected_output=golden["ideal_answer"],
                retrieval_context=context,
                context=reference_context,
            )
        )

    if not test_cases:
        print("No valid test cases available for evaluation.")
        return {}

    metrics = [
        ContextualRecallMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
        ContextualPrecisionMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            run_async=False,
            max_concurrent=1,
        ),
        hyperparameters={
            "pipeline": "agentic_rag",
            "embedding_model": EMBEDDING_MODEL,
            "retrieval_k": RETRIEVAL_K,
            "rerank_top_n": RERANK_TOP_N,
            "judge_model": judge.get_model_name(),
        },
    )

    return summarize_by_metric(result)


def run_local():
    agent = rag_backend.build_agent(pdf_path=PDF_PATH)
    return run(agent)


if __name__ == "__main__":
    print_summary("rag_pipeline", run_local())
