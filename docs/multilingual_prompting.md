# Multilingual Prompting System

The EMSE-Babel evaluation pipeline supports fully multilingual LLM-as-Judge prompting. When evaluating code comments in a non-English language, the entire prompt (taxonomy, instructions, output format) is delivered in the target language.

## Supported Languages

| Language | Code | Prompt Module | Taxonomy File |
|----------|------|---------------|---------------|
| English | `en` | `prompt_english.py` | `error_taxonomy.json` |
| Polish | `pl` | `prompt_polish.py` | `error_taxonomy_polish.json` |
| Dutch | `nl` | `prompt_dutch.py` | `error_taxonomy_dutch.json` |
| Chinese | `zh` | `prompt_chinese.py` | `error_taxonomy_chinese.json` |
| Greek | `el` | `prompt_greek.py` | `error_taxonomy_greek.json` |

## Usage

```bash
# Run with Polish prompts and Polish data
python run_workflow.py --type standard --language Polish --num 50

# The --language flag automatically:
# 1. Loads the Polish prompt library
# 2. Uses prepared_data_Polish/ for input data
# 3. Loads the Polish taxonomy
```

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐
│  run_workflow   │────▶│  PromptLibrary       │
│  --language X   │     │  (language-specific) │
└─────────────────┘     └──────────┬───────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
┌───────────────┐      ┌───────────────────┐      ┌──────────────────┐
│ Taxonomy JSON │      │ System/Assignment │      │ Output Format    │
│ (translated   │      │ Prompts           │      │ (translated JSON │
│  field names) │      │ (target language) │      │  field names)    │
└───────────────┘      └───────────────────┘      └──────────────────┘
                                   │
                                   ▼
                        ┌───────────────────┐
                        │   LLM Response    │
                        │ (target language) │
                        └─────────┬─────────┘
                                  │
                                  ▼
                        ┌───────────────────┐
                        │  ResponseParser   │
                        │  _translate_keys  │
                        │  (→ English)      │
                        └───────────────────┘
```

## What Gets Translated

### 1. Taxonomy Field Names
Each language uses native field names in the taxonomy JSON:

| English | Polish | Dutch | Chinese | Greek |
|---------|--------|-------|---------|-------|
| `name` | `nazwa` | `naam` | `名称` | `ονομα` |
| `description` | `opis` | `beschrijving` | `描述` | `περιγραφη` |
| `inclusion` | `kryteria_wlaczenia` | `inclusiecriteria` | `包含条件` | `κριτηρια_συμπεριληψης` |
| `exclusion` | `kryteria_wylaczenia` | `exclusiecriteria` | `排除条件` | `κριτηρια_εξαιρεσης` |

### 2. Output JSON Format
The expected output structure uses translated field names:

**Polish example:**
```json
{
  "ewaluacje": [
    {
      "nazwa_modelu": "model-name",
      "bledy": ["SE-MD"],
      "wyjasnienie": "...",
      "ogolna_jakosc": "poprawne"
    }
  ]
}
```

### 3. Quality Labels
| English | Polish | Dutch | Chinese | Greek |
|---------|--------|-------|---------|-------|
| `correct` | `poprawne` | `correct` | `正确` | `σωστο` |
| `partially_correct` | `czesciowo_poprawne` | `gedeeltelijk_correct` | `部分正确` | `μερικως_σωστο` |
| `incorrect` | `niepoprawne` | `incorrect` | `不正确` | `λαθος` |

## Response Parsing

The `ResponseParser` in `core/parsing.py` automatically translates non-English responses back to English for aggregation:

1. **Field Mapping**: `FIELD_MAPPINGS` dict maps translated field names → English
2. **Quality Mapping**: `QUALITY_MAPPINGS` dict maps translated quality values → English
3. **`_translate_keys()`**: Recursively translates all keys before normalization

This ensures downstream scoring and analysis always works with a consistent English schema.

## File Structure

```
core/prompting/
├── prompt_base.py      # Base class with shared utilities
├── prompt_english.py   # English (default)
├── prompt_polish.py    # Polish translations
├── prompt_dutch.py     # Dutch translations
├── prompt_chinese.py   # Chinese translations
└── prompt_greek.py     # Greek translations

taxonomy/
├── error_taxonomy.json         # English (source)
├── error_taxonomy_polish.json  # Polish
├── error_taxonomy_dutch.json   # Dutch
├── error_taxonomy_chinese.json # Chinese
└── error_taxonomy_greek.json   # Greek

prompts_lookup/          # Generated prompt previews
├── English/
├── Polish/
├── Dutch/
├── Chinese/
└── Greek/
```

## Adding a New Language

1. **Create taxonomy translation**: Copy `error_taxonomy.json` to `error_taxonomy_{lang}.json`, translate all text values and field names

2. **Create prompt module**: Copy `prompt_english.py` to `prompt_{lang}.py`, update:
   - `LANGUAGE_CODE` class attribute
   - `get_output_field_names()` with translated field names
   - All prompt text methods
   - `output_basic()` and `output_with_reasoning()` with translated JSON schema

3. **Update parser**: Add field mappings to `FIELD_MAPPINGS` and `QUALITY_MAPPINGS` in `core/parsing.py`

4. **Update run_workflow.py**: Add language code to `_lang_map`

5. **Test**: Run `python test_prompts_display.py --language NewLang --all-workflows` to verify

## Generating Prompt Previews

```bash
# Generate all workflows for all languages
python test_prompts_display.py --all

# Generate for specific language
python test_prompts_display.py --language Polish --all-workflows

# Generate specific workflow
python test_prompts_display.py --workflow cot --language Chinese
```

Output files are saved to `prompts_lookup/{Language}/generated_prompt_{workflow}_{Language}.txt`
