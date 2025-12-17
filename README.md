# feedback_llm

A minimal, async feedback pipeline for quantitative social science papers.

## How it works

**Generation**: Scalable teams of specialized AI workers (theorists, rival researchers, methodologists, editors) review the text and propose high-impact feedback. The system uses a balanced ratio of personas (e.g., 3 theorists for every 1 editor) regardless of team size.

**Scoring**: Each proposal is scored twice under different prompt/order variants (rubric order and context order perturbations), then averaged to reduce judge bias.

**Critique & Revision**: Top proposals receive critiques from a "discussant" layer, which then revises the proposals. Revised proposals are re-scored and used for final selection.

**Synthesis**: The highest-quality feedback is synthesized into a structured meta-review containing an **Executive Summary** (prioritized headlines) and a **Technical Implementation Plan** (diagnostic steps).

## Quick Start

### 1. Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Setup API Key (One-time)

Create a file named `.env` in this folder and paste your OpenAI API key inside:

```text
OPENAI_API_KEY=sk-123456789...
```

(The tool loads this automatically. You can also export the key in your terminal if you prefer.)

### 3. Run the Tool (Easiest Way)

1. Create a file named `paper.txt` in this folder.
2. Paste your paper text (e.g., from Overleaf) into it and save.
3. Run:

```bash
python -m feedback_pipeline
```

The tool will find `paper.txt`, generate feedback, and print the meta-review to your screen.

## Advanced Usage & Customization

You can customize the pipeline scale, model power, and synthesis depth using CLI flags.

> **⚠️ Cost Warning**: Using many agents (e.g., 32+) can be very costly, especially with premium models like `gpt-5.2`. Each agent makes API calls, so costs scale linearly with the number of agents. The tool prints cost estimates by default after each run. Start with fewer agents (8-16) and cheaper models (`gpt-5-mini`) to test.

### Customizing Scale & Models

```bash
# Power Run: Use 32 agents and the stronger GPT-5.2 model
python -m feedback_pipeline --agents 32 --model gpt-5.2 --file paper.txt

# Budget Run: Use 8 agents (default) and the cheaper GPT-5-mini
python -m feedback_pipeline --model gpt-5-mini --file paper.txt
```

### All Available Flags

| Flag | Description | Default | Notes |
| --- | --- | --- | --- |
| `--file` | Path to input text file. | `paper.txt` |  |
| `--paste` | Paste text directly into terminal. |  | Cannot use with `--file`. |
| `--agents` | Number of generation workers. | `8` | **Must be a multiple of 8** (e.g., 16, 24, 32). |
| `--model` | OpenAI model to use. | `gpt-5` | Allowed: `gpt-5.2`, `gpt-5.1`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`. |
| `--top-k` | Number of proposals to synthesize. | `5` | Increase this if using high agent counts (e.g., 10 for 32 agents). |
| `--no-cost-estimate` | Skip printing the cost estimate. |  | Cost estimate is printed by default. |

### Example: The "Deep Review"

For a major submission, run a large-scale critique and synthesize the top 10 insights:

```bash
python -m feedback_pipeline --file paper.txt --agents 32 --model gpt-5.2 --top-k 10
```
