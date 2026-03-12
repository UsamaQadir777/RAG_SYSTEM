"""
LLM-based answer generation.
- Clean 1-2 sentence direct answers
- Citations (source documents)
- Confidence scoring
- OpenAI or smart template fallback
"""
import logging
from config import USE_OPENAI, OPENAI_API_KEY, LLM_MODEL, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are answering questions using retrieved document context.

Instructions:
1. Carefully read ALL provided context sections.
2. Find the information that directly answers the question.
3. Ignore irrelevant text such as disclaimers, notices, or unrelated policy sections.
4. Extract the exact answer from the relevant context.
5. Respond in 1-2 sentences only.
6. If the answer is not present in the context, say: "The answer is not found in the provided documents."

Format:
Answer: <short direct answer>
Sources: <document name>"""


def _compute_confidence(results):
    if not results:
        return "low"
    avg_distance = sum(r["distance"] for r in results) / len(results)
    if avg_distance < CONFIDENCE_THRESHOLD * 0.5:
        return "high"
    elif avg_distance < CONFIDENCE_THRESHOLD:
        return "medium"
    else:
        return "low"


def _format_citations(results):
    seen = set()
    citations = []
    for r in results:
        source = r["source"]
        if source not in seen:
            seen.add(source)
            citations.append(source)
    return citations


def _get_source_names(results):
    """Extract clean filenames from source paths."""
    names = []
    for r in results:
        name = r["source"].replace("\\", "/").split("/")[-1]
        if name not in names:
            names.append(name)
    return names


def _build_user_prompt(query, results):
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"[Source {i}: {r['source']}]\n{r['text']}")
    context = "\n\n".join(context_parts)
    return f"Context:\n{context}\n\nQuestion: {query}"


def generate_answer(query, results):
    confidence = _compute_confidence(results)
    citations = _format_citations(results)

    if not results:
        return {
            "answer": "The answer is not found in the provided documents.",
            "citations": [],
            "confidence": "low",
        }

    if USE_OPENAI:
        answer = _generate_with_openai(query, results)
    else:
        answer = _generate_clean_answer(query, results)

    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
    }


def _generate_with_openai(query, results):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(query, results)},
            ],
            temperature=0.1,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()

        # Strip "Answer:" prefix and "Sources:" suffix if LLM added them
        if raw.startswith("Answer:"):
            raw = raw.split("Answer:", 1)[1].strip()
        if "Sources:" in raw:
            raw = raw.split("Sources:")[0].strip()

        return raw

    except Exception as e:
        logger.error("OpenAI generation failed: %s", e)
        return _generate_clean_answer(query, results)


def _generate_clean_answer(query, results):
    """
    Smart extraction when no LLM is available.
    Finds the 1-2 sentences that best answer the question.
    """
    all_text = " ".join(r["text"] for r in results)

    # Normalize: remove markdown headers, extra whitespace, bullet dashes
    clean = all_text.replace("###", "").replace("##", "").replace("#", "")
    clean = clean.replace("\n", " ")
    clean = " ".join(clean.split())

    # Split into sentences
    raw_sentences = []
    for part in clean.replace("- ", ". ").split("."):
        s = part.strip()
        if len(s) > 20:
            raw_sentences.append(s)

    # Score sentences by query keyword overlap
    query_words = set(word.lower() for word in query.split() if len(word) > 2)
    scored = []
    for sentence in raw_sentences:
        sentence_words = set(sentence.lower().split())
        overlap = len(query_words & sentence_words)
        if overlap > 0:
            scored.append((overlap, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = [s for _, s in scored[:2]]

    if best:
        answer_text = ". ".join(best)
        if not answer_text.endswith("."):
            answer_text += "."
        return answer_text                     # ← JUST the answer text, no prefix
    else:
        return "The answer is not found in the provided documents."


def generate_answer_stream(query, results):
    if not results:
        yield "The answer is not found in the provided documents."
        return

    if USE_OPENAI:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)

            stream = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(query, results)},
                ],
                temperature=0.1,
                max_tokens=150,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

        except Exception as e:
            logger.error("OpenAI streaming failed: %s", e)
            yield _generate_clean_answer(query, results)
    else:
        answer = _generate_clean_answer(query, results)
        words = answer.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")