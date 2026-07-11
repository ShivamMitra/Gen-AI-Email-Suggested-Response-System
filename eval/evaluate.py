"""
evaluate.py — THE CORE OF THIS PROJECT.

Defines what "accurate" means for a suggested email reply, and computes it.

============================================================================
WHAT DOES "ACCURATE" MEAN FOR A GENERATED REPLY?
============================================================================
Exact text match against the historical reply is the wrong target: two
replies can use completely different words and both be excellent (or one
could closely mimic the wording of a bad old reply). Instead we decompose
"accuracy" into three independently-checkable dimensions, each answering a
different question a reviewer would actually ask:

  1. RELEVANCE / SEMANTIC SIMILARITY  ("Is this on-topic and in a similar
     spirit to how we've handled this before?")
     -> cosine similarity between TF-IDF vectors of the generated reply and
        the nearest historical reply for a similar incoming email.
     -> Cheap, fast, deterministic, no external dependency -> good for a
        first-pass / regression-safe signal.

  2. KEY-POINT COVERAGE  ("Did it actually address everything the sender
     needed / that a correct reply must contain?")
     -> Each dataset example has a hand-authored `key_points` list — the
        minimal semantic checklist a correct reply must satisfy (e.g.
        "apologizes", "gives refund timeline", "references order number").
     -> We check coverage using keyword/phrase matching against each key
        point (see KEY_POINT_PATTERNS). This is the metric we trust most,
        because it is directly checking task-completion, not just style.
     -> Trade-off: keyword matching is not perfect NLU (a paraphrase could
        be missed). We mitigate this with pattern lists per key point
        rather than single keywords, and by combining with metric 3 below.

  3. LLM-AS-JUDGE (optional, requires ANTHROPIC_API_KEY)  ("Would a human
     reviewer consider this a good, safe, sendable reply?")
     -> Asks Claude to score the reply 1-5 on correctness, tone, and
        completeness, with a short justification. This catches things
        keyword-matching and TF-IDF can't (hallucinated facts, wrong tone,
        factually wrong promises).
     -> Trade-off: costs an API call per evaluation, is non-deterministic,
        and inherits any biases/blind spots of the judge model. We treat it
        as a *complementary* signal, not the sole source of truth — see
        "validation" below for why.

COMPOSITE SCORE (0-100) = weighted combination:
    45% key-point coverage  (task completion — matters most)
    25% semantic similarity (on-topic / stylistically appropriate)
    30% LLM-judge score, rescaled to 0-100 (holistic human-like check)
    (if no API key available, we redistribute the LLM-judge weight
     proportionally across the other two so the score is still 0-100)

We report this per response AND averaged across the dataset as an overall
system score, plus a breakdown by category (some intents are harder than
others, e.g. complaint_escalation vs. informational).

============================================================================
HOW DO WE VALIDATE THE METRIC REFLECTS REAL QUALITY (NOT JUST A NUMBER)?
============================================================================
validate_against_human_ratings() in this file runs a small held-out
"human-labeled" sanity check: we hand-author 3 deliberately-graded replies
per test case (a GOOD reply, a MEDIOCRE reply, and a BAD reply) with our
own 1-5 human quality rating, then compute the Spearman correlation between
our composite score and the human rating. If our metric doesn't rank
good > mediocre > bad consistently, that's a signal the metric needs work.
This is a lightweight stand-in for the "validate against human judgment"
step you'd do at larger scale with real reviewer labels.
"""
import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from statistics import mean

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


# ---------------------------------------------------------------------------
# Metric 1: semantic similarity (TF-IDF cosine) vs. the reference reply
# ---------------------------------------------------------------------------
def semantic_similarity(generated_reply: str, reference_reply: str) -> float:
    vec = TfidfVectorizer(stop_words="english")
    try:
        matrix = vec.fit_transform([generated_reply, reference_reply])
    except ValueError:
        return 0.0
    sim = cosine_similarity(matrix[0], matrix[1])[0][0]
    return float(sim)


# ---------------------------------------------------------------------------
# Metric 2: key-point coverage
# ---------------------------------------------------------------------------
# Pattern lists let us catch common paraphrasings of each key point without
# needing a full NLU model. Extend these as the dataset grows.
KEY_POINT_PATTERNS = {
    "apologizes for the delay": [r"sorry", r"apolog"],
    "apologizes": [r"sorry", r"apolog"],
    "apologizes sincerely": [r"sorry", r"apolog"],
    "genuinely apologizes": [r"sorry", r"apolog"],
    "acknowledges the reschedule request": [r"no problem", r"understand", r"reschedul", r"move"],
    "acknowledges the decision gracefully": [r"understand", r"appreciate", r"no worries"],
    "acknowledges repeated issue": [r"repeat", r"again", r"multiple", r"third time|again"],
    "acknowledges the behavior was unacceptable": [r"not (the|our) standard", r"unacceptable", r"not okay|not ok"],
    "confirms the appointment": [r"confirm"],
    "confirms/proposes a specific time": [r"\d", r"am|pm|monday|tuesday|wednesday|thursday|friday"],
    "proposes/confirms a specific time": [r"\d", r"am|pm|monday|tuesday|wednesday|thursday|friday"],
    "proposes/confirms Friday": [r"friday"],
    "confirms the proposed week": [r"september|week|confirm"],
    "gives specific availability": [r"\d|monday|tuesday|wednesday|thursday|friday|available"],
    "states availability/no conflicts": [r"no conflict|available|works"],
    "gives status/timeline update": [r"transit|day|week|status|update"],
    "gives a status/timeline update": [r"transit|day|week|status|update"],
    "gives timeline": [r"day|week|business day"],
    "gives refund timeline": [r"day|business day"],
    "confirms refund processed": [r"refund"],
    "confirms the double charge/refund": [r"refund|charge"],
    "reassures it won't recur": [r"prevent|won't happen|again|flag"],
    "confirms cancellation": [r"cancel"],
    "confirms no further billing": [r"won't be billed|no (further|more) (charge|billing)"],
    "takes an action (reset account)": [r"reset"],
    "gives troubleshooting step": [r"clear|cache|try|browser"],
    "offers escalation path": [r"escalat"],
    "thanks the customer": [r"thank"],
    "thanks them for the opportunity": [r"thank"],
    "thanks them for interest": [r"thank"],
    "thanks them": [r"thank"],
    "gives a status (roadmap/no plans/etc.)": [r"roadmap|plan|working on|coming"],
    "gives pricing info or range": [r"\$|price|cost"],
    "proposes next step (call)": [r"call|meeting|chat"],
    "grants/addresses the extension request": [r"extend|extension"],
    "confirms new duration": [r"\d+ day|week"],
    "offers further help": [r"help|let me know|happy to"],
    "addresses the comparison directly": [r"compar"],
    "gives concrete differentiators": [r"offer|integrat|support|price"],
    "offers a next step": [r"call|sheet|send|happy to"],
    "leaves door open without being pushy": [r"future|down the road|check back|reach out"],
    "confirms willingness to demo": [r"demo"],
    "gives specific time options": [r"\d|am|pm|monday|tuesday|wednesday|thursday|friday"],
    "flexible/open to alternatives": [r"or another|suggest|works for you"],
    "approves/addresses the specific dates": [r"\d|works|fine|approve"],
    "checks for conflicts": [r"conflict"],
    "reminds about handoff": [r"handoff|hand off|before you leave"],
    "gives enrollment dates": [r"\d|november|start"],
    "answers the re-enrollment question": [r"re-enroll|roll over|automatic"],
    "practical suggestion": [r"log in|confirm|update|check"],
    "welcomes them": [r"welcome"],
    "lists concrete prep items": [r"bring|id|badge|laptop"],
    "reassures rest is covered": [r"orientation|covered|rest"],
    "apologizes/acknowledges delay": [r"sorry|apolog|delay"],
    "gives status": [r"status|approved|queue|in transit"],
    "agrees to help": [r"happy to|glad to|of course|sure"],
    "asks for needed details": [r"send|share|details|could you"],
    "commits to the deadline": [r"deadline|before|by"],
    "escalates/takes concrete action": [r"escalat"],
    "commits to a follow-up timeline": [r"\d+\s*hour|within|follow up"],
    "commits to internal follow-up": [r"share|escalat|review|coaching"],
    "offers to resolve the underlying issue": [r"resolve|help|fix"],
    "takes it seriously / does not dismiss": [r"seriously|apolog|understand"],
    "escalates to appropriate team": [r"team|escalat"],
    "commits to the 48-hour timeline": [r"48"],
    "gives specific rate limit numbers": [r"\d+\s*(requests|rpm|per)"],
    "mentions upgrade path": [r"pro|upgrade|plan"],
    "directly answers open/closed": [r"open|closed"],
    "gives hours if relevant": [r"\d.*(am|pm)|hours"],
    "confirms Canada shipping": [r"canada|ship"],
    "gives cost": [r"\$"],
    "gives delivery time": [r"day|business day"],
    "confirms downtime expected": [r"downtime|down"],
    "gives time window/duration": [r"\d.*(am|pm|min)"],
    "clarifies user action needed": [r"no action|need to|nothing"],
    "gives return window (30 days)": [r"30 day|days"],
    "conditions (unworn/tags)": [r"tag|unworn|condition"],
    "mentions refund/exchange options": [r"refund|exchange"],
    "references order #4521": [r"order"],
    "references order #7788": [r"order"],
    "references Q3 roadmap": [r"q3|roadmap"],
    "references the date/time": [r"\d"],
    "offers to send invite/agenda": [r"invite|agenda|calendar"],
    "polite closing": [r"thank|appreciate|regards|best"],
    "flexible tone": [r"happy|glad|flexible|whichever"],
    "gives a remedy (discount/next step)": [r"discount|refund|credit|next step"],
    "offers a remedy (discount/next step)": [r"discount|refund|credit|next step"],
}


def key_point_coverage(generated_reply: str, key_points: List[str]) -> Dict:
    text = generated_reply.lower()
    covered, missed = [], []
    for kp in key_points:
        patterns = KEY_POINT_PATTERNS.get(kp)
        hit = False
        if patterns:
            hit = any(re.search(p, text) for p in patterns)
        else:
            # fallback: naive substring check on significant words in the key point
            words = [w for w in re.findall(r"[a-z]+", kp.lower()) if len(w) > 4]
            hit = any(w in text for w in words) if words else False
        (covered if hit else missed).append(kp)
    score = len(covered) / len(key_points) if key_points else 1.0
    return {"score": score, "covered": covered, "missed": missed}


# ---------------------------------------------------------------------------
# Metric 3: LLM-as-judge (optional)
# ---------------------------------------------------------------------------
def llm_judge(incoming_email: str, generated_reply: str, key_points: List[str],
              model: str = "llama-3.3-70b-versatile") -> Optional[Dict]:
    """Uses the Groq API as an LLM judge. Requires GROQ_API_KEY."""
    if not os.environ.get("GROQ_API_KEY"):
        return None
    try:
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        prompt = f"""You are grading a suggested email reply for quality.

Incoming email:
{incoming_email}

Key points a correct reply should cover: {key_points}

Suggested reply:
{generated_reply}

Score the reply from 1 (bad) to 5 (excellent) on:
- correctness (does it address the email correctly, no invented facts)
- tone (professional, appropriate)
- completeness (covers the key points)

Respond ONLY with JSON: {{"score": <1-5 integer>, "justification": "<one sentence>"}}"""
        resp = client.chat.completions.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r"^```json|```$", "", text.strip()).strip()
        return json.loads(text)
    except Exception as e:
        return {"score": None, "justification": f"judge_error: {e}"}


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    id: str
    category: str
    composite_score: float  # 0-100
    semantic_similarity: float
    coverage_score: float
    covered_points: List[str]
    missed_points: List[str]
    llm_judge_score: Optional[float]
    llm_judge_justification: Optional[str]
    explanation: str = ""


def evaluate_response(example: Dict, generated_reply: str, use_llm_judge: bool = True) -> EvalResult:
    sim = semantic_similarity(generated_reply, example["reply"])
    cov = key_point_coverage(generated_reply, example.get("key_points", []))

    judge = llm_judge(example["incoming_email"], generated_reply, example.get("key_points", [])) if use_llm_judge else None
    judge_score = None
    judge_just = None
    judge_error = None
    if judge:
        if judge.get("score") is not None:
            judge_score = judge["score"]
            judge_just = judge.get("justification")
        else:
            judge_error = judge.get("justification")  # e.g. "judge_error: <exception>"

    if judge_score is not None:
        composite = 0.45 * cov["score"] * 100 + 0.25 * sim * 100 + 0.30 * (judge_score / 5) * 100
    else:
        # redistribute the 30% judge weight proportionally over the other two
        composite = (0.45 / 0.70) * cov["score"] * 100 + (0.25 / 0.70) * sim * 100

    explanation_bits = [
        f"coverage {cov['score']*100:.0f}% ({len(cov['covered'])}/{len(cov['covered'])+len(cov['missed'])} key points)",
        f"semantic similarity {sim*100:.0f}%",
    ]
    if judge_score is not None:
        explanation_bits.append(f"LLM judge {judge_score}/5 ({judge_just})")
    elif judge_error is not None:
        explanation_bits.append(f"LLM judge FAILED ({judge_error})")
    elif use_llm_judge:
        explanation_bits.append("LLM judge skipped (no GROQ_API_KEY set)")
    else:
        explanation_bits.append("LLM judge disabled for this run")
    if cov["missed"]:
        explanation_bits.append(f"missed: {cov['missed']}")

    return EvalResult(
        id=example["id"],
        category=example["category"],
        composite_score=round(composite, 1),
        semantic_similarity=round(sim, 3),
        coverage_score=round(cov["score"], 3),
        covered_points=cov["covered"],
        missed_points=cov["missed"],
        llm_judge_score=judge_score,
        llm_judge_justification=judge_just,
        explanation="; ".join(explanation_bits),
    )


def evaluate_dataset(results: List[Dict], use_llm_judge: bool = True) -> Dict:
    """
    results: list of {"example": <dataset item>, "generated_reply": str}
    Returns per-response results + overall summary.
    """
    per_response = []
    for r in results:
        er = evaluate_response(r["example"], r["generated_reply"], use_llm_judge=use_llm_judge)
        per_response.append(er)

    overall = mean(r.composite_score for r in per_response) if per_response else 0.0
    by_category: Dict[str, List[float]] = {}
    for r in per_response:
        by_category.setdefault(r.category, []).append(r.composite_score)
    category_summary = {c: round(mean(s), 1) for c, s in by_category.items()}

    return {
        "overall_score": round(overall, 1),
        "n": len(per_response),
        "category_breakdown": category_summary,
        "per_response": [r.__dict__ for r in per_response],
    }


# ---------------------------------------------------------------------------
# Validation: does the metric actually track human-perceived quality?
# ---------------------------------------------------------------------------
def validate_against_human_ratings() -> Dict:
    """
    Small self-authored sanity test: for a few incoming emails we write a
    GOOD, MEDIOCRE, and BAD reply with our own human rating (1-5), then
    check whether the composite score ranks them in the same order.
    This does NOT replace real human evaluation at scale, but demonstrates
    the metric behaves sensibly and gives a quantitative correlation number
    (Spearman-like: fraction of correctly ordered pairs).
    """
    dataset = load_dataset()
    by_id = {d["id"]: d for d in dataset}

    test_cases = [
        {
            "id": "support_001",
            "variants": [
                ("good", 5, "I'm so sorry for the delay with order #4521. I checked and it's in transit, arriving within 3 business days, and I've applied a 10% discount for the trouble."),
                ("mediocre", 3, "Your order is on its way and should arrive soon. Let us know if you have questions."),
                ("bad", 1, "We are not responsible for shipping delays caused by the carrier."),
            ],
        },
        {
            "id": "complaint_001",
            "variants": [
                ("good", 5, "I'm truly sorry we've let you down three times this month - that's not okay. I've escalated your account to our senior team and will follow up within 24 hours with a concrete plan."),
                ("mediocre", 3, "Sorry to hear that. We'll look into it and get back to you at some point."),
                ("bad", 1, "Please try restarting the app, that usually fixes most issues."),
            ],
        },
        {
            "id": "sched_001",
            "variants": [
                ("good", 5, "Wednesday 2pm works great for the Q3 roadmap sync - I'll send a calendar invite with an agenda shortly."),
                ("mediocre", 3, "Sure, let's find a time this week."),
                ("bad", 1, "I don't think we need a Q3 roadmap meeting at all."),
            ],
        },
    ]

    rows = []
    for case in test_cases:
        example = by_id[case["id"]]
        for label, human_score, text in case["variants"]:
            er = evaluate_response(example, text, use_llm_judge=False)
            rows.append({
                "example_id": case["id"], "variant": label,
                "human_score": human_score, "composite_score": er.composite_score,
            })

    # Correctly-ordered-pairs correlation: for every pair within the same
    # example_id, check whether higher human_score -> higher composite_score.
    correct, total = 0, 0
    by_example: Dict[str, List[Dict]] = {}
    for row in rows:
        by_example.setdefault(row["example_id"], []).append(row)
    for ex_id, group in by_example.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a["human_score"] == b["human_score"]:
                    continue
                total += 1
                human_order = a["human_score"] > b["human_score"]
                metric_order = a["composite_score"] > b["composite_score"]
                if human_order == metric_order:
                    correct += 1

    agreement = correct / total if total else None
    return {"rows": rows, "pairwise_agreement": agreement, "correct": correct, "total": total}


if __name__ == "__main__":
    print("=== Validation: metric vs. hand-labeled good/mediocre/bad replies ===")
    val = validate_against_human_ratings()
    for row in val["rows"]:
        print(f"  {row['example_id']:15s} {row['variant']:9s} human={row['human_score']} composite={row['composite_score']}")
    print(f"\nPairwise ranking agreement with human labels: {val['pairwise_agreement']*100:.0f}% "
          f"({val['correct']}/{val['total']} pairs correctly ordered)")