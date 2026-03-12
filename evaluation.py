"""
Evaluation framework for the RAG system.
- 30 test questions (factual, multi-hop, tricky)
- Measures: grounding, hallucination, citation presence
- Outputs structured evaluation report

Usage:
    python evaluation.py
"""
import json
import time
from datetime import datetime, timezone

from rag_pipeline import retrieve, reset
from llm_generator import generate_answer

# --- 30 Test Questions ---
TEST_QUESTIONS = [
    # ── Factual (direct lookup) ──
    {"id": 1, "type": "factual", "question": "What is the company's annual leave policy?"},
    {"id": 2, "type": "factual", "question": "What are the standard working hours?"},
    {"id": 3, "type": "factual", "question": "What is the dress code policy?"},
    {"id": 4, "type": "factual", "question": "How many sick days are employees entitled to?"},
    {"id": 5, "type": "factual", "question": "What is the probation period for new employees?"},
    {"id": 6, "type": "factual", "question": "What is the company's remote work policy?"},
    {"id": 7, "type": "factual", "question": "How does the company handle overtime?"},
    {"id": 8, "type": "factual", "question": "What benefits does the company offer?"},
    {"id": 9, "type": "factual", "question": "What is the policy on workplace harassment?"},
    {"id": 10, "type": "factual", "question": "How are performance reviews conducted?"},

    # ── Multi-hop (requires combining info) ──
    {"id": 11, "type": "multi-hop", "question": "If I use all my sick days, can I use annual leave for medical reasons?"},
    {"id": 12, "type": "multi-hop", "question": "Can a remote worker request overtime pay?"},
    {"id": 13, "type": "multi-hop", "question": "How does the probation period affect leave entitlement?"},
    {"id": 14, "type": "multi-hop", "question": "What happens if a harassment complaint is filed during probation?"},
    {"id": 15, "type": "multi-hop", "question": "Can performance review outcomes impact remote work eligibility?"},
    {"id": 16, "type": "multi-hop", "question": "How do working hours differ for part-time vs full-time employees?"},
    {"id": 17, "type": "multi-hop", "question": "What is the process if I want to extend my probation period and take leave?"},
    {"id": 18, "type": "multi-hop", "question": "Does the dress code apply when working remotely?"},
    {"id": 19, "type": "multi-hop", "question": "How are benefits affected during extended sick leave?"},
    {"id": 20, "type": "multi-hop", "question": "Can overtime during probation count toward performance evaluation?"},

    # ── Tricky (edge cases, out-of-scope, adversarial) ──
    {"id": 21, "type": "tricky", "question": "What is the CEO's favorite color?"},
    {"id": 22, "type": "tricky", "question": "Can I bring my pet to the office?"},
    {"id": 23, "type": "tricky", "question": "What is the company's stock price?"},
    {"id": 24, "type": "tricky", "question": "Tell me about the company's competitors."},
    {"id": 25, "type": "tricky", "question": "What will the leave policy be next year?"},
    {"id": 26, "type": "tricky", "question": "Is the company planning layoffs?"},
    {"id": 27, "type": "tricky", "question": "Summarize every policy in one sentence."},
    {"id": 28, "type": "tricky", "question": "What if I disagree with the leave policy?"},
    {"id": 29, "type": "tricky", "question": "Can I override the dress code with manager approval?"},
    {"id": 30, "type": "tricky", "question": "What is the meaning of life?"},
]


def evaluate():
    """Run all test questions through the RAG pipeline and evaluate."""
    reset()  # Fresh state

    results = []
    total_latency = 0

    print("=" * 70)
    print("RAG SYSTEM EVALUATION REPORT")
    print(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Total Questions: {len(TEST_QUESTIONS)}")
    print("=" * 70)

    for tq in TEST_QUESTIONS:
        start = time.perf_counter()

        # Retrieve
        chunks = retrieve(tq["question"])

        # Generate
        response = generate_answer(tq["question"], chunks)

        latency_ms = (time.perf_counter() - start) * 1000
        total_latency += latency_ms

        # Evaluation metrics
        has_citations = len(response["citations"]) > 0
        answer_text = response["answer"].lower()

        # Check if answer is grounded (mentions context/documents)
        grounded = (
            has_citations
            and response["confidence"] != "low"
            and "don't have enough information" not in answer_text
        )

        # Check for potential hallucination signals
        hallucination_signals = [
            "i think",
            "probably",
            "i believe",
            "as far as i know",
            "generally speaking",
        ]
        potential_hallucination = any(
            signal in answer_text for signal in hallucination_signals
        )

        entry = {
            "id": tq["id"],
            "type": tq["type"],
            "question": tq["question"],
            "answer": response["answer"][:200],
            "confidence": response["confidence"],
            "citations": response["citations"],
            "has_citations": has_citations,
            "grounded": grounded,
            "potential_hallucination": potential_hallucination,
            "latency_ms": round(latency_ms, 2),
        }
        results.append(entry)

        status = "✅" if grounded and not potential_hallucination else "⚠️"
        print(
            f"\n{status} Q{tq['id']:02d} [{tq['type']:<9}] {tq['question'][:55]}"
        )
        print(
            f"   Confidence: {response['confidence']} | "
            f"Citations: {has_citations} | "
            f"Grounded: {grounded} | "
            f"Hallucination: {potential_hallucination} | "
            f"{latency_ms:.0f} ms"
        )

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for qtype in ["factual", "multi-hop", "tricky"]:
        subset = [r for r in results if r["type"] == qtype]
        grounded_count = sum(1 for r in subset if r["grounded"])
        halluc_count = sum(1 for r in subset if r["potential_hallucination"])
        cited_count = sum(1 for r in subset if r["has_citations"])
        avg_lat = (
            sum(r["latency_ms"] for r in subset) / len(subset) if subset else 0
        )

        print(f"\n  {qtype.upper()} ({len(subset)} questions):")
        print(f"    Grounded:      {grounded_count}/{len(subset)}")
        print(f"    Hallucination: {halluc_count}/{len(subset)}")
        print(f"    Has Citations: {cited_count}/{len(subset)}")
        print(f"    Avg Latency:   {avg_lat:.0f} ms")

    print(f"\n  OVERALL:")
    print(f"    Total Latency:  {total_latency:.0f} ms")
    print(f"    Avg Latency:    {total_latency / len(results):.0f} ms")
    print("=" * 70)

    # Save report
    report_path = "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    evaluate()