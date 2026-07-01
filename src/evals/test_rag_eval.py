import os

os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "300")
os.environ.setdefault("DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE", "300")

import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from src.app.rag_chain import RAGPipeline

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

JUDGE_MODEL_NAME = os.getenv("DEEPEVAL_JUDGE_MODEL", "qwen2.5:14b")

IS_CI = os.getenv("CI") == "true" or "1b" in JUDGE_MODEL_NAME.lower()
METRIC_THRESHOLD = 0.5 if IS_CI else 0.7

judge_model = OllamaModel(
    model=JUDGE_MODEL_NAME,
    base_url=OLLAMA_URL,
    temperature=0.0,
)

rag = RAGPipeline()


def build_test_case(query: str) -> LLMTestCase:
    result = rag.query(query)
    return LLMTestCase(
        input=query,
        actual_output=result["answer"],
        retrieval_context=result["context"],
    )


def run_metrics(test_case: LLMTestCase, metrics: list) -> None:
    assert_test(test_case, metrics, run_async=False)


def test_rag_faithfulness_rag_definition():
    """Validates that RAG definition answers remain grounded in the knowledge base."""
    test_case = build_test_case("What is Retrieval-Augmented Generation?")

    faithfulness_metric = FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=judge_model)
    relevancy_metric = AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge_model)
    run_metrics(test_case, [faithfulness_metric, relevancy_metric])


def test_rag_faithfulness_llm_challenges():
    """Validates that answers explaining LLM challenges accurately match retrieved text."""
    test_case = build_test_case("What are the known challenges of using raw LLMs?")

    faithfulness_metric = FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=judge_model)
    run_metrics(test_case, [faithfulness_metric])


def test_rag_faithfulness_rag_benefits():
    """Validates that RAG benefit explanations are accurate to the knowledge source."""
    test_case = build_test_case("What are the primary benefits of implementing RAG?")

    faithfulness_metric = FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=judge_model)
    relevancy_metric = AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge_model)
    run_metrics(test_case, [faithfulness_metric, relevancy_metric])