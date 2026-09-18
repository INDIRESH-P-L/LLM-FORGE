#!/usr/bin/env python3
import sys
import py_compile
from pathlib import Path

def test_syntax():
    files = [
        "scripts/05_rag.py",
        "app/main.py",
        "app/chat_api.py",
    ]
    for f in files:
        py_compile.compile(f, doraise=True)
        print(f"[OK] Syntax valid: {f}")

def test_continuation_detection():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rag_module",
        str(Path("scripts/05_rag.py")),
    )
    rag_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag_mod)

    # Test is_continuation_query
    assert rag_mod.is_continuation_query("continue") is True
    assert rag_mod.is_continuation_query("continue please") is True
    assert rag_mod.is_continuation_query("please continue") is True
    assert rag_mod.is_continuation_query("go on") is True
    assert rag_mod.is_continuation_query("proceed") is True
    assert rag_mod.is_continuation_query("tell me more") is True
    assert rag_mod.is_continuation_query("Can bail be denied under PMLA?") is False

    # Test is_short_followup_query
    assert rag_mod.is_short_followup_query("what are the exceptions?") is True
    assert rag_mod.is_short_followup_query("why?") is True

    # Test extract_conversation_context
    sample_history = [
        {
            "role": "user",
            "content": "Can bail be denied under PMLA if the twin conditions of Section 45 are not met? Cite recent SC judgments."
        },
        {
            "role": "assistant",
            "content": "Section 45 of PMLA sets forth twin conditions for bail..."
        }
    ]

    last_q, last_a, retr_q = rag_mod.extract_conversation_context("continue", sample_history)
    assert last_q == "Can bail be denied under PMLA if the twin conditions of Section 45 are not met? Cite recent SC judgments."
    assert "Section 45 of PMLA" in last_a
    assert "Section 45 PMLA" in retr_q

    # Test build_rag_prompt with continuation
    prompt = rag_mod.build_rag_prompt(
        query="continue",
        chunks=[{"citation_id": 1, "source": "PMLA", "title": "Section 45", "text": "Twin conditions"}],
        citations=["[Source 1] Section 45 PMLA"],
        is_continuation=True,
        last_user_query=last_q,
        last_assistant_response=last_a,
    )
    assert "SPECIAL INSTRUCTION FOR CONTINUATION:" in prompt
    assert "CONTINUATION, NOT DEFINITION" in prompt
    assert "Manish Sisodia" in prompt
    assert "Vijay Madanlal Choudhary" in prompt

    print("[OK] All continuation unit tests passed successfully!")

if __name__ == "__main__":
    test_syntax()
    test_continuation_detection()
