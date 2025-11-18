import asyncio
import json
import sys
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple

import tiktoken
from openai import AsyncOpenAI


"""
Very lightweight async feedback pipeline.

High-level steps:
1. Generate proposals (8 independent workers).
2. Score each proposal (4 criteria).
3. Rank and classify proposals in Python only.
4. Produce a meta-review from all high-quality proposals.

Details of prompts / thresholds are intentionally minimal for now
and can be refined later.
"""


GENERATION_MODEL = "gpt-5.1-mini"
SCORING_MODEL = "gpt-5.1-mini"
META_MODEL = "gpt-5.1"
N_GENERATION_WORKERS = 8

MODEL_PRICING = {
    # Prices in USD per token (converted from USD per million tokens)
    "gpt-5.1": {
        "input": 1.25 / 1_000_000,
        "output": 10.0 / 1_000_000,
        "cached_input": 0.125 / 1_000_000,
    },
    "gpt-5.1-mini": {
        "input": 0.25 / 1_000_000,
        "output": 2.0 / 1_000_000,
        "cached_input": 0.025 / 1_000_000,
    },
}

_ENCODER_CACHE: Dict[str, tiktoken.Encoding] = {}


def _encoding_for_model(model: str) -> tiktoken.Encoding:
    encoding = _ENCODER_CACHE.get(model)
    if encoding is not None:
        return encoding
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    _ENCODER_CACHE[model] = encoding
    return encoding


def _count_text_tokens(text: str, model: str) -> int:
    return len(_encoding_for_model(model).encode(text))


def _count_message_tokens(messages: List[Dict[str, str]], model: str) -> int:
    return sum(_count_text_tokens(message["content"], model) for message in messages)


def _generation_user_prompt(paper_text: str, worker_id: int) -> str:
    return f"""
You review the paper text below and provide exactly one feedback proposal.

Note: The text you receive may be only part of the full manuscript (for example, just the introduction or methods). Work with whatever material is provided and surface the most impactful feedback you can. You may briefly acknowledge missing context, but do not ask for more text.

Your goal is to identify the single most important change that would improve the paper.
Choose one of the following dimensions that best fits your feedback:
- "contribution": novelty, substantive importance, positioning in the literature.
- "logical_soundness": logical coherence and internal consistency of the argument.
- "interpretation": interpretation of empirical results and their connection to theory.
- "writing_structure": clarity of exposition, organization, and structure.

Requirements for the feedback proposal:
- Maximum 3 sentences.
- Focus on the single most important issue you see.
- Reference at least one concrete element of the text (for example, a section, claim, figure, or type of analysis).
- Use neutral, precise, and technical language.
- Make the proposal directly actionable if possible.
- Do not request additional sections; instead, provide the best possible guidance given the excerpt.

Return a JSON object with fields:
- "id": integer worker id ({worker_id})
- "dimension": one of ["contribution","logical_soundness","interpretation","writing_structure"]
- "text": the feedback text

Paper text:
```text
{paper_text}
```""".strip()


GENERATION_SYSTEM_PROMPT = (
    "You are an expert reviewer for quantitative social science papers. "
    "You produce a single, concise, high-impact feedback proposal."
)


def _generation_messages(paper_text: str, worker_id: int) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": _generation_user_prompt(paper_text, worker_id)},
    ]


def _scoring_user_prompt(
    paper_text: str,
    proposal_text: str,
    proposal_dimension: str,
) -> str:
    return f"""
You receive the paper text and one feedback proposal.

Assign four integer scores from 1 to 5:
- "importance": impact of the feedback on improving the paper.
- "specificity": degree of grounding in concrete, identifiable parts of the text.
- "actionability": clarity of what the author should change based on this feedback.
- "uniqueness": distinctiveness relative to typical comments on such papers.

Return a JSON object:
- "importance": integer 1–5
- "specificity": integer 1–5
- "actionability": integer 1–5
- "uniqueness": integer 1–5

Paper text:
```text
{paper_text}

Feedback proposal (dimension = {proposal_dimension}):

{proposal_text}
```""".strip()


SCORING_SYSTEM_PROMPT = (
    "You evaluate the quality of a single feedback proposal for a social science paper. "
    "You assign integer scores only."
)


def _scoring_messages(
    paper_text: str,
    proposal_text: str,
    proposal_dimension: str,
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SCORING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _scoring_user_prompt(
                paper_text,
                proposal_text,
                proposal_dimension,
            ),
        },
    ]


META_SYSTEM_PROMPT = (
    "You are writing a concise meta-review that synthesizes selected "
    "feedback proposals on a quantitative social science paper."
)


def _meta_messages(selection: Dict[str, Any]) -> List[Dict[str, str]]:
    by_dim_payload = {
        dim: [
            {
                "id": p["id"],
                "text": p["text"],
                "importance": p["importance"],
                "specificity": p["specificity"],
                "actionability": p["actionability"],
                "uniqueness": p["uniqueness"],
                "composite": p["composite"],
            }
            for p in plist
        ]
        for dim, plist in selection["by_dimension"].items()
    }

    top_global = selection["sorted"][:TOP_K]
    top_global_payload = [
        {
            "id": p["id"],
            "dimension": p["dimension"],
            "text": p["text"],
            "composite": p["composite"],
        }
        for p in top_global
    ]

    user_content = f"""
You receive high-quality feedback proposals grouped by dimension, and the globally strongest proposals.

Dimensions:
- contribution
- logical_soundness
- interpretation
- writing_structure

Write a meta-review with four sections, in this order:
1. Contribution
2. Logical soundness of the argument
3. Interpretation of empirical results
4. Writing and structure

For each section:
- If there are proposals for that dimension, write 2–3 sentences that integrate their content and provide directly actionable guidance.
- If there are no proposals for that dimension, write 1–2 sentences indicating that no major issues were flagged there.

Then provide a numbered list of the three most important revisions across all dimensions, ordered from most to least important. Base this list primarily on the globally strongest proposals, but you may merge or rephrase them for clarity.

High-quality proposals by dimension:
```json
{json.dumps(by_dim_payload)}

Globally strongest proposals:
{json.dumps(top_global_payload)}
```""".strip()

    return [
        {"role": "system", "content": META_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

client = AsyncOpenAI()  # Requires OPENAI_API_KEY in environment


# Thresholds for meta-review inclusion (tune as needed)
IMPORTANCE_THRESHOLD = 3
COMPOSITE_THRESHOLD = 3.0
TOP_K = 3

DIMENSIONS = [
    "contribution",
    "logical_soundness",
    "interpretation",
    "writing_structure",
]


# -------------------------------------------------------------------
# Helper: generic JSON chat call
# -------------------------------------------------------------------


async def chat_json(
    messages: List[Dict[str, str]],
    model: str = GENERATION_MODEL,
) -> Any:
    """Call the chat API and parse a JSON object response."""
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    return json.loads(content)


# -------------------------------------------------------------------
# 1. Independent generation workers
# -------------------------------------------------------------------


async def generate_single_proposal(paper_text: str, worker_id: int) -> Dict[str, Any]:
    messages = _generation_messages(paper_text, worker_id)
    result = await chat_json(messages)
    result["id"] = worker_id  # enforce id
    return result


async def generate_all_proposals(
    paper_text: str,
    n_workers: int = N_GENERATION_WORKERS,
) -> List[Dict[str, Any]]:
    tasks = [
        generate_single_proposal(paper_text, worker_id=i)
        for i in range(1, n_workers + 1)
    ]
    proposals = await asyncio.gather(*tasks)
    return proposals


# -------------------------------------------------------------------
# 2. Independent scoring workers
# -------------------------------------------------------------------


async def score_single_proposal(
    paper_text: str,
    proposal: Dict[str, Any],
) -> Dict[str, Any]:
    messages = _scoring_messages(
        paper_text,
        proposal.get("text", ""),
        proposal.get("dimension", ""),
    )

    scores = await chat_json(messages, model=SCORING_MODEL)
    scored = {
        **proposal,
        "importance": int(scores["importance"]),
        "specificity": int(scores["specificity"]),
        "actionability": int(scores["actionability"]),
        "uniqueness": int(scores["uniqueness"]),
    }
    return scored


async def score_all_proposals(
    paper_text: str,
    proposals: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    tasks = [score_single_proposal(paper_text, p) for p in proposals]
    scored = await asyncio.gather(*tasks)

    # Compute composite deterministically in Python
    for s in scored:
        I = s["importance"]
        S = s["specificity"]
        A = s["actionability"]
        U = s["uniqueness"]
        composite = 0.4 * I + 0.3 * S + 0.2 * A + 0.1 * U
        s["composite"] = composite

    return scored


# -------------------------------------------------------------------
# 3. Deterministic selection, ranking, thresholds
# -------------------------------------------------------------------


def select_and_classify(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Sort by composite, descending
    sorted_scored = sorted(scored, key=lambda x: x["composite"], reverse=True)

    # Top K
    top_proposals = sorted_scored[:TOP_K]

    # Low-value proposals
    low_value_ids = [
        p["id"]
        for p in sorted_scored
        if p["importance"] <= 2 or p["actionability"] <= 2
    ]

    # High-quality proposals for meta-review
    high_quality = [
        p
        for p in sorted_scored
        if (p["composite"] >= COMPOSITE_THRESHOLD)
        or (p["importance"] >= IMPORTANCE_THRESHOLD)
    ]

    # Group high-quality proposals by dimension
    by_dimension = {dim: [] for dim in DIMENSIONS}
    for p in high_quality:
        dim = p.get("dimension")
        if dim in by_dimension:
            by_dimension[dim].append(p)

    selection = {
        "sorted": sorted_scored,
        "top_proposals": top_proposals,
        "low_value_ids": low_value_ids,
        "high_quality": high_quality,
        "by_dimension": by_dimension,
    }
    return selection


# -------------------------------------------------------------------
# 4. Meta-review using all high-quality proposals
# -------------------------------------------------------------------


async def meta_review(selection: Dict[str, Any]) -> str:
    messages = _meta_messages(selection)
    resp = await client.chat.completions.create(
        model=META_MODEL,
        messages=messages,
    )
    return resp.choices[0].message.content


# -------------------------------------------------------------------
# 5. Cost estimation helpers (tiktoken-based)
# -------------------------------------------------------------------


def _stage_cost_summary(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
) -> Dict[str, Any]:
    pricing = MODEL_PRICING.get(model)
    cost = None
    if pricing:
        cost = (
            prompt_tokens * pricing["input"]
            + completion_tokens * pricing["output"]
        )
    total_tokens = prompt_tokens + completion_tokens
    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost,
    }


def _estimate_generation_tokens(
    paper_text: str,
    proposals: List[Dict[str, Any]],
) -> Tuple[int, int]:
    prompt_tokens = 0
    completion_tokens = 0
    for proposal in proposals:
        worker_id = proposal.get("id", 0)
        messages = _generation_messages(paper_text, worker_id)
        prompt_tokens += _count_message_tokens(messages, GENERATION_MODEL)
        completion_tokens += _count_text_tokens(
            json.dumps(
                {
                    "id": proposal.get("id"),
                    "dimension": proposal.get("dimension"),
                    "text": proposal.get("text"),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            GENERATION_MODEL,
        )
    return prompt_tokens, completion_tokens


def _estimate_scoring_tokens(
    paper_text: str,
    scored: List[Dict[str, Any]],
) -> Tuple[int, int]:
    prompt_tokens = 0
    completion_tokens = 0
    for proposal in scored:
        messages = _scoring_messages(
            paper_text,
            proposal.get("text", ""),
            proposal.get("dimension", ""),
        )
        prompt_tokens += _count_message_tokens(messages, SCORING_MODEL)
        completion_tokens += _count_text_tokens(
            json.dumps(
                {
                    "importance": proposal.get("importance"),
                    "specificity": proposal.get("specificity"),
                    "actionability": proposal.get("actionability"),
                    "uniqueness": proposal.get("uniqueness"),
                },
                separators=(",", ":"),
            ),
            SCORING_MODEL,
        )
    return prompt_tokens, completion_tokens


def _estimate_meta_tokens(
    selection: Dict[str, Any],
    meta_review_text: str,
) -> Tuple[int, int]:
    if not selection:
        return 0, _count_text_tokens(meta_review_text, META_MODEL)
    messages = _meta_messages(selection)
    prompt_tokens = _count_message_tokens(messages, META_MODEL)
    completion_tokens = _count_text_tokens(meta_review_text, META_MODEL)
    return prompt_tokens, completion_tokens


def estimate_pipeline_cost(
    paper_text: str,
    pipeline_output: Dict[str, Any],
) -> Dict[str, Any]:
    proposals = pipeline_output.get("proposals", [])
    scored = pipeline_output.get("scored", [])
    selection = pipeline_output.get("selection", {})
    meta_review_text = pipeline_output.get("meta_review", "")

    gen_prompt, gen_completion = _estimate_generation_tokens(paper_text, proposals)
    score_prompt, score_completion = _estimate_scoring_tokens(paper_text, scored)
    meta_prompt, meta_completion = _estimate_meta_tokens(selection, meta_review_text)

    stages = {
        "generation": _stage_cost_summary(gen_prompt, gen_completion, GENERATION_MODEL),
        "scoring": _stage_cost_summary(score_prompt, score_completion, SCORING_MODEL),
        "meta_review": _stage_cost_summary(meta_prompt, meta_completion, META_MODEL),
    }

    total_prompt_tokens = sum(stage["prompt_tokens"] for stage in stages.values())
    total_completion_tokens = sum(stage["completion_tokens"] for stage in stages.values())
    total_cost = sum((stage["cost_usd"] or 0.0) for stage in stages.values())

    return {
        "stages": stages,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "total_cost_usd": total_cost,
    }


# -------------------------------------------------------------------
# 6. Full pipeline wrapper + convenience entry point
# -------------------------------------------------------------------


async def full_feedback_pipeline(paper_text: str) -> Dict[str, Any]:
    """Run the full async feedback pipeline for a single paper."""
    proposals = await generate_all_proposals(paper_text)
    scored = await score_all_proposals(paper_text, proposals)
    selection = select_and_classify(scored)
    meta = await meta_review(selection)

    result = {
        "proposals": proposals,
        "scored": scored,
        "selection": selection,
        "meta_review": meta,
    }
    result["cost_estimate"] = estimate_pipeline_cost(paper_text, result)
    return result


def feedback(paper_text: str) -> str:
    """
    Synchronous convenience wrapper.

    Returns only the meta-review text. For more detailed inspection
    (scores, selection, etc.), use `full_feedback_pipeline` directly.
    """
    return asyncio.run(full_feedback_pipeline(paper_text))["meta_review"]


__all__ = [
    "full_feedback_pipeline",
    "feedback",
    "generate_all_proposals",
    "score_all_proposals",
    "select_and_classify",
    "meta_review",
    "estimate_pipeline_cost",
]
@@
    "meta_review",
]


def _read_paper_from_stdin(prompt: bool = False) -> str:
    if prompt:
        print(
            "Paste paper text, then press Ctrl-D (Ctrl-Z then Enter on Windows) when finished:\n",
            file=sys.stderr,
            end="",
            flush=True,
        )
    return sys.stdin.read()


def _read_paper_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _format_cost_estimate(cost: Dict[str, Any]) -> str:
    lines = []
    for stage_name, summary in cost.get("stages", {}).items():
        cost_usd = summary.get("cost_usd")
        cost_str = f"${cost_usd:.4f}" if cost_usd is not None else "n/a"
        lines.append(
            f"- {stage_name}: prompt={summary.get('prompt_tokens', 0)}, "
            f"completion={summary.get('completion_tokens', 0)}, cost≈{cost_str}"
        )
    total_cost = cost.get("total_cost_usd")
    total_cost_str = f"${total_cost:.4f}" if total_cost is not None else "n/a"
    lines.append(
        f"- total: prompt={cost.get('total_prompt_tokens', 0)}, "
        f"completion={cost.get('total_completion_tokens', 0)}, "
        f"cost≈{total_cost_str}"
    )
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    """
    Minimal CLI entry point.

    Usage examples:
      python -m feedback_pipeline --file paper.txt
      cat paper.txt | python -m feedback_pipeline
    """
    parser = ArgumentParser(description="Run the feedback pipeline on a paper.")
    parser.add_argument(
        "--file",
        "-f",
        help="Path to a text file containing the paper. If omitted, read from stdin.",
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Print a tiktoken-based cost estimate after running the pipeline.",
    )
    parser.add_argument(
        "--paste",
        action="store_true",
        help="Prompt for interactive paste when no --file is provided.",
    )
    args = parser.parse_args(argv)

    if args.file and args.paste:
        parser.error("--paste cannot be used together with --file")

    if args.file:
        paper_text = _read_paper_from_file(args.file)
    else:
        prompt_for_paste = args.paste or sys.stdin.isatty()
        paper_text = _read_paper_from_stdin(prompt_for_paste)

    if not paper_text.strip():
        print("No paper text provided (file was empty or stdin had no content).", file=sys.stderr)
        return 1

    result = asyncio.run(full_feedback_pipeline(paper_text))
    print(result["meta_review"])

    if args.estimate_cost:
        cost = result.get("cost_estimate")
        if cost:
            print("\n---\nApproximate token usage (tiktoken)")
            print(_format_cost_estimate(cost))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
