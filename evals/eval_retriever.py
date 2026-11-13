
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
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
MAX_EVAL_CASES = 4

def run(retriever):
    goldens = load_goldens(GOLDEN_PATH)[:MAX_EVAL_CASES]
    judge = GroqEvaluationLLM()

    test_cases = []

    for golden in goldens:
        documents = retriever.invoke(golden["query"])

        retrieved_context = [
            doc.page_content
            for doc in documents
            if doc.page_content
        ]

        if not retrieved_context:
            print(f"Skipped empty retrieval: {golden['query']}")
            continue

        reference_context = golden["reference_context"]

        if isinstance(reference_context, str):
            reference_context = [reference_context]

        test_cases.append(
            LLMTestCase(
                input=golden["query"],
                actual_output="Retriever evaluation",
                expected_output=golden["ideal_answer"],
                retrieval_context=retrieved_context,
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
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            run_async=True,
            max_concurrent=2,
        ),
        hyperparameters={
            "retriever": "multi_query_flashrank",
            "embedding_model": EMBEDDING_MODEL,
            "retrieval_k": RETRIEVAL_K,
            "rerank_top_n": RERANK_TOP_N,
            "judge_model": judge.get_model_name(),
        },
    )

    return summarize_by_metric(result)


def run_local():
    rag_backend.build_agent(pdf_path=PDF_PATH)
    retriever = rag_backend.retriever

    return run(retriever)


if __name__ == "__main__":
    print_summary("retriever", run_local())
