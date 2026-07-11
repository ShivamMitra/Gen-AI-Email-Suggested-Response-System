"""
run_pipeline.py — end-to-end demo.

For every email in the dataset, treats it as a "new incoming email",
generates a suggested reply (retrieving from the REST of the dataset, i.e.
leave-one-out so we're not just retrieving the exact same example), then
evaluates the generated reply against that example's reference reply and
key_points checklist.

Prints per-response scores + an overall report, and writes a JSON report to
report.json.

Run:
    python run_pipeline.py                 # offline mode (no API key)
    GROQ_API_KEY=gsk_... python run_pipeline.py   # full Gen-AI mode (Groq)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads GROQ_API_KEY from a .env file in the project root, if present
except ImportError:
    pass  # python-dotenv not installed -> falls back to relying on real env vars only

from generator import load_dataset, Retriever, build_prompt, call_groq, offline_fallback_generate
from evaluate import evaluate_dataset


def generate_leave_one_out(dataset, k=3, model="llama-3.3-70b-versatile"):
    use_llm = bool(os.environ.get("GROQ_API_KEY"))
    results = []
    error_count = 0
    for i, example in enumerate(dataset):
        rest = dataset[:i] + dataset[i + 1:]
        retriever = Retriever(rest)
        examples = retriever.top_k(example["subject"], example["incoming_email"], k=k)

        if use_llm:
            try:
                prompt = build_prompt(example["subject"], example["incoming_email"], examples)
                reply = call_groq(prompt, model=model)
            except Exception as e:
                error_count += 1
                if error_count <= 3:  # don't spam the console for every one of 29 examples
                    print(f"  [WARN] Gemini call failed for '{example['id']}', falling back to offline. Error: {e}")
                reply = offline_fallback_generate(example["subject"], example["incoming_email"], examples)
        else:
            reply = offline_fallback_generate(example["subject"], example["incoming_email"], examples)

        results.append({"example": example, "generated_reply": reply})

    if use_llm and error_count > 0:
        print(f"  [WARN] {error_count}/{len(dataset)} Gemini calls failed and used the offline fallback instead. "
              f"Check the error message above (invalid key, wrong model name, quota, or network).")
    return results


def main():
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} dataset examples.")
    use_llm_gen = bool(os.environ.get("GROQ_API_KEY"))
    print(f"Generation mode: {'LLM (Groq API)' if use_llm_gen else 'offline fallback (no GROQ_API_KEY)'}")
    print("Generating suggested replies (leave-one-out retrieval)...")
    gen_results = generate_leave_one_out(dataset)

    print("Evaluating...")
    use_llm_judge = bool(os.environ.get("GROQ_API_KEY"))
    report = evaluate_dataset(gen_results, use_llm_judge=use_llm_judge)

    print("\n" + "=" * 70)
    print("PER-RESPONSE SCORES")
    print("=" * 70)
    for r in report["per_response"]:
        print(f"[{r['id']:15s}] score={r['composite_score']:5.1f}/100  {r['explanation']}")

    print("\n" + "=" * 70)
    print("CATEGORY BREAKDOWN")
    print("=" * 70)
    for cat, score in report["category_breakdown"].items():
        print(f"  {cat:22s} {score:5.1f}/100")

    print("\n" + "=" * 70)
    print(f"OVERALL SYSTEM SCORE: {report['overall_score']}/100  (n={report['n']})")
    print("=" * 70)

    out_path = os.path.join(os.path.dirname(__file__), "report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
