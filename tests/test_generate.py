from copilot.generate import SYSTEM_PROMPT, answer_question, cited_sources, format_context
from copilot.retrieval import Hit
from tests.conftest import SAMPLE_CHUNKS, FakeClaude, text_response

HITS = [Hit(chunk, 0.9 - i * 0.1) for i, chunk in enumerate(SAMPLE_CHUNKS[:3])]


def test_format_context_numbers_excerpts_with_citations() -> None:
    context = format_context(HITS)
    assert context.startswith("[1] Nike_FY2025_10K.pdf, p. 38 (Nike)")
    assert "[3] Columbia_FY2024_10K.pdf, p. 5" in context


def test_cited_sources_renumbers_to_match_source_list() -> None:
    # The model cites excerpts 3 and 2 only; the answer must say [1], [2] to match the list.
    text, sources = cited_sources("Staff [3]. Stores grew [2][3]. Both [2, 3]. Unknown [7].", HITS)
    assert sources == ["Columbia_FY2024_10K.pdf, p. 5", "Lululemon_FY2024_10K.pdf, p. 31"]
    assert text == "Staff [1]. Stores grew [2][1]. Both [2][1]. Unknown ."


def test_answer_question_sends_excerpts_and_returns_citations() -> None:
    fake = FakeClaude([text_response("Gross margin fell 190 bps to 42.7% [1].")])
    answer = answer_question("What happened to Nike's gross margin?", HITS, client=fake, model="test-model")

    request = fake.requests[0]
    assert request["model"] == "test-model"
    assert request["system"] == SYSTEM_PROMPT
    assert "[1] Nike_FY2025_10K.pdf, p. 38" in request["messages"][0]["content"]
    assert answer.sources == ["Nike_FY2025_10K.pdf, p. 38"]
    assert len(answer.contexts) == 3
