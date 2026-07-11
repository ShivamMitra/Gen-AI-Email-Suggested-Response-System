"""
generator.py

Given a new incoming email, produces a suggested reply using:
  1. Retrieval over the dataset (TF-IDF cosine similarity) to find the
     k most similar past (email, reply) pairs -> used as few-shot examples.
  2. A prompt built from those few-shot examples + the new email, sent to
     an LLM (Claude, via the Anthropic API) to generate the reply.

WHY RETRIEVAL + FEW-SHOT PROMPTING (instead of fine-tuning):
  - Dataset is small (tens, not thousands, of examples) - fine-tuning would
    overfit and isn't necessary; a strong general-purpose LLM already knows
    how to write email replies, it just needs *grounding* in this
    inbox/company's tone and facts.
  - Retrieval keeps the system data-driven and inspectable: for any output
    we can point at exactly which past emails informed it, which matters a
    lot for the evaluation system built in eval/.
  - Easy to update: adding a new past-email example immediately changes
    future suggestions with no retraining step.
  - Trade-off: pure prompting can't learn subtle statistical patterns the
    way fine-tuning could at scale, and quality is bounded by how good the
    retrieved examples are. For a production system with thousands of
    emails we'd likely combine this with periodic light fine-tuning /
    preference-tuning on accepted vs. edited suggestions.

If GOOGLE_API_KEY is not set, falls back to a deterministic offline
"generator" that stitches together the single best-matching past reply
with light templating, so the whole pipeline still runs end-to-end without
network access / an API key (useful for grading / CI).

This uses the Gemini API (Google AI Studio) via the `google-generativeai`
SDK. Default model: "gemini-2.5-flash" (fast, cheap, strong enough for
short email replies). Use "gemini-2.5-pro" for higher-quality generation
at higher cost/latency.
"""
import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads GOOGLE_API_KEY from a .env file in the project root, if present
except ImportError:
    pass  # python-dotenv not installed -> falls back to relying on real env vars only

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.jsonl")


def load_dataset(path: str = DATA_PATH) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Retriever:
    """TF-IDF retrieval over subject+body of past emails."""

    def __init__(self, dataset: List[Dict]):
        self.dataset = dataset
        corpus = [f"{d['subject']} {d['incoming_email']}" for d in dataset]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    def top_k(self, subject: str, body: str, k: int = 3) -> List[Dict]:
        q = self.vectorizer.transform([f"{subject} {body}"])
        sims = cosine_similarity(q, self.matrix)[0]
        ranked_idx = sims.argsort()[::-1][:k]
        results = []
        for i in ranked_idx:
            item = dict(self.dataset[i])
            item["_similarity"] = float(sims[i])
            results.append(item)
        return results


def build_prompt(subject: str, body: str, examples: List[Dict]) -> str:
    ex_blocks = []
    for ex in examples:
        ex_blocks.append(
            f"Past incoming email (subject: {ex['subject']}):\n{ex['incoming_email']}\n\n"
            f"Reply that was actually sent:\n{ex['reply']}\n"
        )
    examples_text = "\n---\n".join(ex_blocks)

    prompt = f"""You are an assistant that drafts suggested email replies for a
professional inbox. Below are past incoming emails and the replies that were
actually sent in response to similar situations. Use them ONLY as style and
content guidance (tone, level of detail, what info/actions a good reply
includes) - do not copy them verbatim, and do not invent facts (order
numbers, dates, amounts) that aren't given in the new email.

{examples_text}
---

Now draft a suggested reply to this NEW incoming email. Keep the tone
professional and warm, be concise (3-6 sentences), and directly address
what the sender is asking for. If the email requires information you don't
have (e.g. exact order number, exact time), reply naturally without making
up facts.

New incoming email (subject: {subject}):
{body}

Suggested reply:"""
    return prompt


def call_groq(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """Calls the Groq API. Requires GROQ_API_KEY in the environment."""
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def offline_fallback_generate(subject: str, body: str, examples: List[Dict]) -> str:
    """
    Deterministic, no-network fallback used when no API key is present.
    NOT meant to be as good as the LLM path - it exists purely so the
    pipeline (generation + evaluation) is fully runnable/gradeable offline.
    It lightly adapts the single closest past reply.
    """
    best = examples[0]
    reply = best["reply"]
    # Light templated adaptation: prepend an acknowledgement referencing the
    # new subject so it isn't a pure verbatim copy.
    opener = f"Thanks for your message about \"{subject.strip()}\"."
    return f"{opener} {reply}"


@dataclass
class GenerationResult:
    subject: str
    incoming_email: str
    suggested_reply: str
    retrieved_examples: List[Dict] = field(default_factory=list)
    method: str = "llm"


def generate_reply(subject: str, body: str, k: int = 3, model: str = "llama-3.3-70b-versatile") -> GenerationResult:
    dataset = load_dataset()
    retriever = Retriever(dataset)
    examples = retriever.top_k(subject, body, k=k)

    use_llm = bool(os.environ.get("GROQ_API_KEY"))
    if use_llm:
        try:
            prompt = build_prompt(subject, body, examples)
            reply = call_groq(prompt, model=model)
            method = "llm"
        except Exception as e:  # network / auth / quota issues -> fall back
            reply = offline_fallback_generate(subject, body, examples)
            method = f"offline_fallback (llm_error: {e})"
    else:
        reply = offline_fallback_generate(subject, body, examples)
        method = "offline_fallback (no GROQ_API_KEY set)"

    return GenerationResult(
        subject=subject,
        incoming_email=body,
        suggested_reply=reply,
        retrieved_examples=examples,
        method=method,
    )


if __name__ == "__main__":
    demo_subject = "Order still not here"
    demo_body = "I ordered a laptop stand 10 days ago (order #9931) and tracking hasn't updated in a week. Can you look into this?"
    result = generate_reply(demo_subject, demo_body)
    print("METHOD:", result.method)
    print("\nRETRIEVED EXAMPLES:")
    for ex in result.retrieved_examples:
        print(f"  [{ex['_similarity']:.3f}] {ex['id']} - {ex['subject']}")
    print("\nSUGGESTED REPLY:\n", result.suggested_reply)