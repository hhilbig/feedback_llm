import asyncio
import json
import os
import sys
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple

# --- Smart API Key Loading ---
from dotenv import load_dotenv

# This loads .env only if the variable isn't already set in the environment
load_dotenv()

# Safety check: Catch missing keys early so the error is readable
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY is missing.", file=sys.stderr)
    print("   Please either:", file=sys.stderr)
    print(
        "   1. Create a file named '.env' containing: OPENAI_API_KEY=sk-...",
        file=sys.stderr,
    )
    print("   2. Or export it in your terminal.", file=sys.stderr)
    sys.exit(1)
# ----------------------------------

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


GENERATION_MODEL = "gpt-5"
SCORING_MODEL = "gpt-5"
META_MODEL = "gpt-5.1"

MODEL_PRICING = {
    "gpt-5.2": {
        "input": 1.75 / 1e6,
        "output": 14.00 / 1e6,
        "cached_input": 0.175 / 1e6,
    },
    "gpt-5.1": {
        "input": 1.25 / 1e6,
        "output": 10.00 / 1e6,
        "cached_input": 0.125 / 1e6,
    },
    "gpt-5": {"input": 1.25 / 1e6, "output": 10.00 / 1e6, "cached_input": 0.125 / 1e6},
    "gpt-5-mini": {
        "input": 0.25 / 1e6,
        "output": 2.00 / 1e6,
        "cached_input": 0.025 / 1e6,
    },
    "gpt-5-nano": {
        "input": 0.05 / 1e6,
        "output": 0.40 / 1e6,
        "cached_input": 0.005 / 1e6,
    },
}


def _lookup_pricing_model(model: str) -> Dict[str, float]:
    # Strict lookup only. No prefix matching.
    if model not in MODEL_PRICING:
        raise ValueError(
            f"Model '{model}' is not allowed. Choose from: {list(MODEL_PRICING.keys())}"
        )
    return MODEL_PRICING[model]


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


def _progress(message: str) -> None:
    print(f"[feedback] {message}", file=sys.stderr)


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
- Style: whenever possible, frame your feedback as a constructive, inquisitive question (e.g., "I'm wondering if...")
- Do not request additional sections; instead, provide the best possible guidance given the excerpt.
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
    "You are part of a multidisciplinary review panel for social science manuscripts. "
    "Follow any persona instructions provided to focus your expertise on the most impactful feedback."
)


def _generation_messages(
    persona_prompt: str,
    paper_text: str,
    worker_id: int,
) -> List[Dict[str, str]]:
    system_prompt = persona_prompt or GENERATION_SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
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
- "specificity": degree of grounding in the text.
  - For empirical feedback this means referencing a specific table, result, or method section.
  - For conceptual or logical feedback this means referencing a specific claim, argument, or unstated assumption.
  - Critically, rate conceptual feedback as highly specific (4 or 5) when it pinpoints a real argumentative gap,
    even if the reasoning is abstract.
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


CRITIC_SYSTEM_PROMPT = (
    "You are an expert reviewer acting as a discussant. Your job is to scrutinize a colleague's "
    "feedback, identify flaws or gaps, and suggest substantive improvements. Be concise but candid."
)


def _critic_user_prompt(paper_text: str, proposal: Dict[str, Any]) -> str:
    proposal_json = json.dumps(proposal, ensure_ascii=False, separators=(",", ":"))
    return f"""
You receive the paper text and one high-quality feedback proposal that another reviewer wrote.

Your task is to critique or significantly improve this proposal.
- Point out logical flaws, misinterpretations, or missing context.
- Suggest sharper, more precise guidance when possible.
- If the proposal is already excellent, say so explicitly.

Return a JSON object:
- "original_id": {proposal.get("id")}
- "critique_text": your 1-2 sentence critique or improvement.

Paper text:
```text
{paper_text}
```

Proposal to critique:
```json
{proposal_json}
```""".strip()


def _critic_messages(paper_text: str, proposal: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": _critic_user_prompt(paper_text, proposal)},
    ]


META_SYSTEM_PROMPT = (
    "You are a collegial senior researcher composing first-pass reading notes for the authors. "
    "Adopt an inquisitive, constructive tone—more like a mentoring email than a formal review. "
    "You synthesize the junior specialists' proposals into a clear, actionable set of insights."
)


def _meta_messages(selection: Dict[str, Any], top_k: int) -> List[Dict[str, str]]:
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

    # Use dynamic top_k here
    top_global = selection.get("sorted_by_composite", [])[:top_k]
    top_global_payload = [
        {
            "id": p["id"],
            "dimension": p["dimension"],
            "text": p["text"],
            "composite": p["composite"],
        }
        for p in top_global
    ]

    # Use dynamic top_k here
    unique_payload = [
        {
            "id": p["id"],
            "dimension": p["dimension"],
            "text": p["text"],
            "uniqueness": p["uniqueness"],
            "composite": p["composite"],
        }
        for p in selection.get("sorted_by_uniqueness", [])[:top_k]
    ]

    critiques_payload = selection.get("critiques", [])
    all_high_quality_payload = [
        {
            "id": p["id"],
            "dimension": p["dimension"],
            "text": p["text"],
            "importance": p["importance"],
            "specificity": p["specificity"],
            "actionability": p["actionability"],
            "uniqueness": p["uniqueness"],
            "composite": p["composite"],
        }
        for p in selection.get("high_quality", [])
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

Consider both the globally strongest proposals and the most unique proposals so that high-impact but novel insights are not overlooked.

Before writing the final list, perform an explicit prioritization step:
- Review all high-quality proposals below.
- Balance two goals: (1) conceptual/logical soundness (theory, assumptions, alternative explanations) and (2) empirical validity (identification, statistical interpretation).
- Select the three to five most important revisions (default to three unless additional issues are truly distinct). You MUST include at least one conceptual/logical flaw if any such proposals exist. Do not simply choose the empirically strongest points.

Then provide a numbered list of the prioritized revisions, ordered from most to least important. Base this list primarily on your balanced prioritization, weaving in unique proposals when they surface distinct, valuable issues. Use an inquisitive tone where appropriate.

High-quality proposals by dimension:
```json
{json.dumps(by_dim_payload)}

Globally strongest proposals (by composite score):
{json.dumps(top_global_payload)}

Most unique proposals (by uniqueness score):
{json.dumps(unique_payload)}

Critiques from discussant reviewers:
{json.dumps(critiques_payload)}

All high-quality proposals (for prioritization):
{json.dumps(all_high_quality_payload)}
```

Crucially, for each prioritized revision, include a one-sentence justification explaining why it was prioritized (impact, logical priority, or risk). For example:
1. [Revision text?] *Justification: Addresses a foundational gap in the paper's core argument.*
2. [Revision text?] *Justification: Resolves a critical interpretive error in the main empirical claim.*
""".strip()

    return [
        {"role": "system", "content": META_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


client = AsyncOpenAI()  # Requires OPENAI_API_KEY in environment


# Thresholds for meta-review inclusion (tune as needed)
IMPORTANCE_THRESHOLD = 3
COMPOSITE_THRESHOLD = 3.0

DIMENSIONS = [
    "contribution",
    "logical_soundness",
    "interpretation",
    "writing_structure",
]

PERSONA_THEORIST = (
    "You are a senior social theorist and logician. Your only mandate is to uncover foundational flaws "
    "in the paper's conceptual framework, central argument, and unstated assumptions. Prioritize the "
    "'contribution' and 'logical_soundness' dimensions.\n"
    "CRITICAL INSTRUCTION: You are forbidden from discussing specific econometric techniques, identification "
    "strategies, robustness checks, or statistical diagnostics. Stay entirely focused on the theory, logic, "
    "and framing of the research question.\n"
    "Good example: 'Are these policies really a form of adaptation, or merely delaying pain? The paper should "
    "define this premise more explicitly.'"
)

PERSONA_RIVAL = (
    "You are a rival researcher probing the paper's interpretation. Your job is to surface rival hypotheses, "
    "alternative mechanisms, omitted contextual factors, or selection effects that could also explain the "
    "reported outcomes. Concentrate on the 'interpretation' dimension and avoid dwelling on statistical "
    "implementation details.\n"
    "Good example: 'Could the null effect in high-stress areas simply reflect that residents there already rely "
    "on last-resort insurance plans and thus are insulated from the policy?'"
)

PERSONA_METHODOLOGIST = (
    "You are a quantitative methodologist. Scrutinize empirical design choices, identification clarity, and the "
    "interpretation of statistical evidence. Focus on 'logical_soundness' when it pertains to methods and on "
    "'interpretation' when data usage or diagnostics are at stake. Do not comment on prose quality or high-level theory."
)

PERSONA_EDITOR = (
    "You are a senior journal editor evaluating clarity, organization, and narrative structure. Concentrate on the "
    "'writing_structure' dimension and resist the temptation to critique statistical methods or theoretical framing."
)

BASE_PERSONA_DECK = [
    PERSONA_THEORIST,
    PERSONA_THEORIST,
    PERSONA_THEORIST,  # 3 Theorists
    PERSONA_RIVAL,
    PERSONA_RIVAL,  # 2 Rivals
    PERSONA_METHODOLOGIST,
    PERSONA_METHODOLOGIST,  # 2 Methodologists
    PERSONA_EDITOR,  # 1 Editor
]


def create_worker_assignments(num_agents: int) -> List[Dict[str, Any]]:
    # 1. Validation
    if num_agents <= 0 or num_agents % 8 != 0:
        raise ValueError(
            f"Agent count must be a multiple of 8 (8, 16, 24...). Got {num_agents}."
        )

    # 2. Multiplication
    num_blocks = num_agents // 8
    full_deck = BASE_PERSONA_DECK * num_blocks

    # 3. Assignment Construction
    assignments = []
    for i, persona in enumerate(full_deck):
        assignments.append({"id": i + 1, "persona": persona})
    return assignments


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


async def generate_single_proposal(
    paper_text: str,
    worker_id: int,
    persona_prompt: str,
) -> Dict[str, Any]:
    messages = _generation_messages(persona_prompt, paper_text, worker_id)
    result = await chat_json(messages)
    result["id"] = worker_id  # enforce id
    result["persona"] = persona_prompt
    return result


async def generate_all_proposals(
    paper_text: str,
    workers: List[Dict[str, Any]],  # CHANGED: Now accepts specific worker list
    model: str,  # CHANGED: Now accepts specific model
) -> List[Dict[str, Any]]:

    tasks = []
    for assignment in workers:
        messages = _generation_messages(
            assignment["persona"], paper_text, assignment["id"]
        )

        # We call chat_json explicitly with the chosen model
        task = chat_json(messages, model=model)
        tasks.append(task)

    # Gather results and re-attach IDs (since order is preserved in gather)
    raw_results = await asyncio.gather(*tasks)

    formatted_results = []
    for i, result in enumerate(raw_results):
        worker = workers[i]
        result["id"] = worker["id"]
        result["persona"] = worker["persona"]
        formatted_results.append(result)

    return formatted_results


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
        composite = 0.35 * I + 0.25 * S + 0.20 * A + 0.20 * U
        s["composite"] = composite

    return scored


# -------------------------------------------------------------------
# 3. Deterministic selection, ranking, thresholds
#    + Delphi-style critique round helpers
# -------------------------------------------------------------------


async def critique_single_proposal(
    paper_text: str,
    proposal: Dict[str, Any],
) -> Dict[str, Any]:
    messages = _critic_messages(paper_text, proposal)
    critique = await chat_json(messages, model=GENERATION_MODEL)
    critique["original_id"] = proposal.get("id")
    return critique


async def run_critique_round(
    paper_text: str,
    proposals_to_critique: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not proposals_to_critique:
        return []
    tasks = [critique_single_proposal(paper_text, p) for p in proposals_to_critique]
    critiques = await asyncio.gather(*tasks)
    return critiques


def select_and_classify(scored: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    # Sort by composite, descending
    sorted_by_composite = sorted(scored, key=lambda x: x["composite"], reverse=True)

    # Top K by composite
    top_proposals = sorted_by_composite[:top_k]

    # Low-value proposals
    low_value_ids = [
        p["id"]
        for p in sorted_by_composite
        if p["importance"] <= 2 or p["actionability"] <= 2
    ]

    # High-quality proposals for meta-review
    high_quality = [
        p
        for p in sorted_by_composite
        if (p["composite"] >= COMPOSITE_THRESHOLD)
        or (p["importance"] >= IMPORTANCE_THRESHOLD)
    ]

    # Also rank high-quality proposals by uniqueness to surface novel ideas
    sorted_by_uniqueness = sorted(
        high_quality,
        key=lambda x: x["uniqueness"],
        reverse=True,
    )

    # Group high-quality proposals by dimension
    by_dimension = {dim: [] for dim in DIMENSIONS}
    for p in high_quality:
        dim = p.get("dimension")
        if dim in by_dimension:
            by_dimension[dim].append(p)

    selection = {
        "sorted_by_composite": sorted_by_composite,
        "sorted_by_uniqueness": sorted_by_uniqueness,
        "top_proposals": top_proposals,
        "low_value_ids": low_value_ids,
        "high_quality": high_quality,
        "by_dimension": by_dimension,
    }
    return selection


# -------------------------------------------------------------------
# 4. Meta-review using all high-quality proposals
# -------------------------------------------------------------------


async def meta_review(selection: Dict[str, Any], top_k: int) -> str:
    messages = _meta_messages(selection, top_k)
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
    pricing = _lookup_pricing_model(model)
    cost = prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]
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
        persona_prompt = proposal.get("persona", GENERATION_SYSTEM_PROMPT)
        messages = _generation_messages(persona_prompt, paper_text, worker_id)
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


def _estimate_critique_tokens(
    paper_text: str,
    high_quality: List[Dict[str, Any]],
    critiques: List[Dict[str, Any]],
) -> Tuple[int, int]:
    if not high_quality:
        return 0, 0
    prompt_tokens = 0
    completion_tokens = 0
    critique_by_id = {c.get("original_id"): c for c in critiques}
    for proposal in high_quality:
        messages = _critic_messages(paper_text, proposal)
        prompt_tokens += _count_message_tokens(messages, GENERATION_MODEL)
        critique = critique_by_id.get(proposal.get("id"))
        if critique:
            completion_tokens += _count_text_tokens(
                json.dumps(
                    {
                        "original_id": critique.get("original_id"),
                        "critique_text": critique.get("critique_text"),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                GENERATION_MODEL,
            )
    return prompt_tokens, completion_tokens


def _estimate_meta_tokens(
    selection: Dict[str, Any],
    meta_review_text: str,
    top_k: int,
) -> Tuple[int, int]:
    if not selection:
        return 0, _count_text_tokens(meta_review_text, META_MODEL)
    messages = _meta_messages(selection, top_k)
    prompt_tokens = _count_message_tokens(messages, META_MODEL)
    completion_tokens = _count_text_tokens(meta_review_text, META_MODEL)
    return prompt_tokens, completion_tokens


def estimate_pipeline_cost(
    paper_text: str,
    pipeline_output: Dict[str, Any],
    gen_model: str,  # New Argument
    top_k: int = 5,  # For cost estimation
) -> Dict[str, Any]:
    proposals = pipeline_output.get("proposals", [])
    scored = pipeline_output.get("scored", [])
    selection = pipeline_output.get("selection", {})
    meta_review_text = pipeline_output.get("meta_review", "")

    gen_prompt, gen_completion = _estimate_generation_tokens(paper_text, proposals)
    score_prompt, score_completion = _estimate_scoring_tokens(paper_text, scored)
    critiques = selection.get("critiques", [])
    critique_prompt, critique_completion = _estimate_critique_tokens(
        paper_text,
        selection.get("high_quality", []),
        critiques,
    )
    meta_prompt, meta_completion = _estimate_meta_tokens(
        selection, meta_review_text, top_k
    )

    stages = {
        "generation": _stage_cost_summary(gen_prompt, gen_completion, gen_model),
        "scoring": _stage_cost_summary(score_prompt, score_completion, SCORING_MODEL),
        "critiques": _stage_cost_summary(
            critique_prompt,
            critique_completion,
            gen_model,
        ),
        "meta_review": _stage_cost_summary(meta_prompt, meta_completion, META_MODEL),
    }

    total_prompt_tokens = sum(stage["prompt_tokens"] for stage in stages.values())
    total_completion_tokens = sum(
        stage["completion_tokens"] for stage in stages.values()
    )
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


async def full_feedback_pipeline(
    paper_text: str,
    num_agents: int = 8,
    gen_model: str = "gpt-5",
    top_k: int = 5,  # New default
) -> Dict[str, Any]:
    """Run the full async feedback pipeline for a single paper."""

    # 1. Create workers dynamically
    workers = create_worker_assignments(num_agents)

    _progress(f"Generating proposals with {num_agents} agents using {gen_model}…")
    proposals = await generate_all_proposals(paper_text, workers, gen_model)

    _progress("Scoring proposals…")
    scored = await score_all_proposals(paper_text, proposals)

    _progress(f"Selecting Top-{top_k} feedback…")
    # Pass top_k here
    selection = select_and_classify(scored, top_k)

    _progress("Running critique round…")
    critiques = await run_critique_round(paper_text, selection.get("high_quality", []))
    selection["critiques"] = critiques

    _progress("Writing meta-review…")
    # Pass top_k here
    meta = await meta_review(selection, top_k)

    result = {
        "proposals": proposals,
        "scored": scored,
        "selection": selection,
        "meta_review": meta,
    }
    # Pass gen_model and top_k here for accurate pricing
    result["cost_estimate"] = estimate_pipeline_cost(
        paper_text, result, gen_model, top_k
    )
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


def _read_paper_from_stdin(
    prompt: bool = False,
    sentinel: str | None = None,
) -> str:
    if prompt and sys.stdin.isatty() and sentinel:
        print(
            "Paste paper text below. When you're done, type "
            f"a line containing only {sentinel!r} and press Enter.\n",
            file=sys.stderr,
            end="",
            flush=True,
        )
        lines: List[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == sentinel:
                break
            lines.append(line)
        return "\n".join(lines)

    if prompt and sys.stdin.isatty():
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
        help="Path to a text file containing the paper.",
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Print a tiktoken-based cost estimate after running the pipeline.",
    )
    parser.add_argument(
        "--paste",
        action="store_true",
        help="Prompt for interactive paste (forces paste mode).",
    )
    parser.add_argument(
        "--agents", type=int, default=8, help="Number of agents (must be multiple of 8)"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-5", choices=list(MODEL_PRICING.keys())
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top proposals to include in meta-review",
    )
    args = parser.parse_args(argv)

    if args.file and args.paste:
        parser.error("--paste cannot be used together with --file")

    sentinel = "::END::" if (args.paste or sys.stdin.isatty()) else None

    # --- INPUT LOGIC START ---

    # 1. Explicit file passed via CLI
    if args.file:
        paper_text = _read_paper_from_file(args.file)

    # 2. Piped input (e.g. cat paper.txt | python ...)
    elif not sys.stdin.isatty():
        paper_text = sys.stdin.read()

    # 3. Default file "paper.txt" (The Co-author Friendly Path)
    elif os.path.exists("paper.txt"):
        print("Found 'paper.txt'. Reading from file...", file=sys.stderr)
        paper_text = _read_paper_from_file("paper.txt")

    # 4. Fallback to interactive paste
    else:
        print(
            "No input provided. Paste text below (end with ::END::) OR create 'paper.txt'.",
            file=sys.stderr,
        )
        prompt_for_paste = args.paste or sys.stdin.isatty()
        paper_text = _read_paper_from_stdin(prompt_for_paste, sentinel=sentinel)

    # --- INPUT LOGIC END ---

    if not paper_text.strip():
        print(
            "No paper text provided (file was empty or stdin had no content).",
            file=sys.stderr,
        )
        return 1

    try:
        result = asyncio.run(
            full_feedback_pipeline(
                paper_text,
                num_agents=args.agents,
                gen_model=args.model,
                top_k=args.top_k,
            )
        )
    except ValueError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        return 1

    print(result["meta_review"])

    if args.estimate_cost:
        cost = result.get("cost_estimate")
        if cost:
            print("\n---\nApproximate token usage (tiktoken)")
            print(_format_cost_estimate(cost))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
