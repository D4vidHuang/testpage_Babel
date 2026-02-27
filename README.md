# EMSE-Babel

LLM-as-a-Judge pipeline for evaluating code comment errors against a taxonomy.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenRouter API key
export OPENROUTER_API_KEY="your-key"  # Linux/Mac
$env:OPENROUTER_API_KEY = "your-key"  # PowerShell

# Run evaluation on 30 English instances
python run_workflow.py --type standard --num 30 --score

# Run advanced combined workflow (Workflow 8)
python run_workflow.py --type combined --num 30 --score
```

## Usage

### Run Evaluations

```bash
# Standard evaluation (Workflow 4)
python run_workflow.py --type standard --num 50

# Standard workflow (useful for debugging)
python run_workflow.py --type standard --num 5 --random --verbose

# Chain-of-thought (G-Eval style, Workflow 5)
python run_workflow.py --type cot --num 50

# Hierarchical (category clusters, Workflow 6)
python run_workflow.py --type hierarchical --num 50

# Rubric-based (PRESENT/ABSENT, Workflow 7)
python run_workflow.py --type rubric --num 50

# Combined evaluation (Hierarchical + Rubric + CoT + Bias Mitigation, Workflow 8)
python run_workflow.py --type combined --num 50
```

### Options
, `combined` |
| `--num`, `-n` | Number of instances to evaluate |
| `--data`, `-d` | Input data file (default: English) |
| `--score` | Run scoring after evaluation |
| `--score-only` | Score existing results without running evaluation |
| `--model`, `-m` | Judge model (default: `anthropic/claude-3.5-sonnet`) |
| `--verbose`, `-v` | Show detailed output for each instance |
| `--random`, `-r` | Randomly select instances instead of sequential |
| `--list-types` | List all available workflow types with descriptionsuation |
| `--model`, `-m` | Judge model (default: `anthropic/claude-3.5-sonnet`) |
| `--verbose`, `-v` | Show detailed output for each instance |
| `--random` | Randomly select instances instead of sequential |

### Other Languages

```bash
# Dutch
python run_workflow.py --type standard --data prepared_data_Dutch/final_batch_judge.json --num 30

# Chinese
python run_workflow.py --type standard --data prepared_data_Chinese/final_batch_judge.json --num 30
```

### Score Existing Results

```bash
python run_workflow.py --score-only --pred output/standard_evaluation/output_1/judge_results.json
```
# List Available Workflows

```bash
# See all workflow types with descriptions
python run_workflow.py --list-types
```

## Workflow Types

| Workflow | Type | Description | Key Features |
|----------|------|-------------|--------------|
| **Workflow 4** | `standard` | Basic evaluation with structured JSON output | Simple, fast, baseline approach |
| **Workflow 5** | `cot` | G-Eval style with chain-of-thought reasoning | Explicit reasoning steps, improved accuracy |
| **Workflow 6** | `hierarchical` | Category-clustered evaluation | Reduces cognitive load, 7 semantic clusters |
| **Workflow 7** | `rubric` | Taxonomy as rubric with PRESENT/ABSENT criteria | Clear decision criteria, structured guidance |
| **Workflow 8** | `combined` | All techniques combined | **Bias mitigation + Hierarchical + Rubric + CoT** |

### Workflow 8 (Combined Evaluation)

The most sophisticated evaluation approach, combining all research-based improvements:

1. **Bias Mitigation** - Uses enhanced system prompt (v2) to reduce position and verbosity bias
2. **Hierarchical Clustering** - Evaluates 7_evaluation/output_N/`:

- `judge_results.json` - Model evaluations with consolidated errors
- `run_stats.json` - Parse success rates, API failures, execution metrics

### Output Structure

**Standard workflows (4, 5, 7):**
```json
{
  "instance_id": "60_1",
  "judge_evaluations": {
    "evaluations": [
      {
        "model_name": "Qwen/CodeQwen1.5-7B",
        "errors": ["LG-GR5", "SE-MD"],
        "explanation": "The comment has grammatical issues..."
      }
    ]
  }
}
```

**Hierarchical workflows (6, 8):**
```json
{
  "instance_id": "60_1",
  "judge_evaluations": {
    "evaluations": [...]
  },
  "cluster_details": [
    {
      "cluster_name": "linguistic_grammar",
      "error_ids": ["LG-GR1", "LG-GR2", ...],
      "raw_response": "Full reasoning text...",
      "parsed_evaluations": [...]
    }
  ]
}
```

The `cluster_details` field preserves cluster-level reasoning for research analysis while maintaining scorer compatibility.
**Trade-offs:**
- ✅ Highest accuracy and interpretability
- ✅ Detailed cluster-level analysis preserved
- ⚠️ 7× more API calls than standard workflow
- ⚠️ Longer execution time

**Example:**
```bash
# Run on 30 instances with verbose output
python run_workflow.py --type combined --num 30 --verbose

# Run and score immediately
python run_workflow.py --type combined --num 50 --score
```

##
## Project Structure

```
├── run_workflow.py      # Main CLI entry point
├── core/                # Core modules
│   ├── judge.py         # OpenRouter/Ollama API wrapper
│   ├── config.py        # JudgeConfig dataclass
│   ├── manager.py       # Orchestration logic
│   ├── prompts.py       # Prompt templates
│   ├── parsing.py       # Response parsing
│   └── data.py          # Data loading/export
├── scoring/             # Scoring module
│   └── scorer.py        # Metrics computation
├── taxonomy/            # Error taxonomy JSON
├── prepared_data_*/     # Evaluation data by language
├── output/              # Evaluation results
└── tests/               # Test suite
```

## Output

Results are saved to `output/<workflow_type>/output_N/`:

- `judge_results.json` - Model evaluations with raw responses
- `run_stats.json` - Parse success rates, API failures, token usage

### Run Statistics Metrics

The `run_stats.json` file tracks:

- **Schema Compliance Rate**: Percentage of responses following the expected JSON schema without warnings
- **Parse Success Rate**: Percentage of responses successfully parsed (includes those with warnings)
- **API Success Rate**: Percentage of successful API calls
- **Token Usage**: Input, thinking (for supported models), and output tokens

## Score Analysis

Analyze results with the all-in-one notebook:

```bash
cd scoring
jupyter notebook analysis.ipynb
```

Set `RUN_PATH` in cell 4 (e.g., `"output/standard_evaluation/output_22"`), then run all cells to:
- Aggregate evaluation results by instance-model pairs
- Merge with neural metrics (BARTScore)
- Generate visualizations: run statistics, error analysis, quality distributions
- Export aggregated data to `aggregated_data/` and plots to `visualization_output_*/`


## Tests

```bash
python -m pytest tests/ -v
```