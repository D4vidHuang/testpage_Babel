#!/usr/bin/env python3
"""Generate workflow prompts for inspection without making API calls.

Generates and saves the complete prompts that would be sent to the judge model
for different workflow types and languages.

Usage (from project root):
    # Generate prompt for a specific workflow and language
    python prompts_lookup/generate_prompts.py --workflow standard --language English
    
    # Generate all workflows for a language
    python prompts_lookup/generate_prompts.py --language English --all-workflows
    
    # Generate all workflows for all languages
    python prompts_lookup/generate_prompts.py --all
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports when run from prompts_lookup/
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import DataLoader
from run_workflow import WORKFLOW_TYPES


def build_prompt_messages(workflow_type: str, PromptLib, cluster_name=None, cluster_taxonomy=None):
    """Build the system and user messages for a workflow using the specified PromptLib."""
    wf = WORKFLOW_TYPES[workflow_type]
    
    # Select system prompt
    if wf["system_prompt"] == "enhanced":
        system_prompt = PromptLib.system_enhanced()
    else:
        system_prompt = PromptLib.system_basic()
    
    # Build assignment message
    if cluster_name and cluster_taxonomy:
        # Hierarchical or Combined: use cluster-specific taxonomy
        taxonomy_str = json.dumps(cluster_taxonomy, indent=2, ensure_ascii=False)
        assignment = PromptLib.assignment_evaluate_cluster(cluster_name, taxonomy_str)
        # For combined workflow: add rubric formatting to clusters
        if wf.get("use_rubric"):
            assignment += "\n\n" + PromptLib.format_taxonomy_as_rubric(cluster_taxonomy)
    elif workflow_type == "rubric":
        # Rubric: use formatted rubric (language-specific)
        taxonomy = PromptLib.load_taxonomy()
        assignment = PromptLib.assignment_evaluate_models(taxonomy)
        assignment += "\n\n" + PromptLib.format_taxonomy_as_rubric(taxonomy)
    else:
        # Standard/CoT: use JSON taxonomy (language-specific)
        taxonomy = PromptLib.load_taxonomy()
        taxonomy_str = json.dumps(taxonomy, indent=2, ensure_ascii=False)
        assignment = PromptLib.assignment_evaluate_models(taxonomy_str)
    
    # Add output instruction
    if wf["output_instruction"] == "reasoning":
        assignment += "\n\n" + PromptLib.output_with_reasoning()
    else:
        assignment += "\n\n" + PromptLib.output_basic()
    
    return system_prompt, assignment


def simulate_workflow(workflow_type: str, language: str, num_instances: int = 1, output_file: str = None):
    """Simulate a workflow and save the complete prompt to a file.
    
    Args:
        workflow_type: Type of workflow (standard, cot, rubric, hierarchical, combined)
        language: Language for prompts (English, Polish, Chinese, Dutch, Greek)
        num_instances: Number of instances to include in simulation (default: 1)
        output_file: Output filename (default: generated_prompt_{workflow}_{language}.txt)
    
    Returns:
        Path to the generated file
    """
    # Default output filename in language-specific directory
    if output_file is None:
        output_dir = Path("prompts_lookup") / language
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"generated_prompt_{workflow_type}_{language}.txt"
        output_file = str(output_file)
    
    # Build output content
    output_lines = []
    
    output_lines.append("=" * 80)
    output_lines.append(f"WORKFLOW PROMPT SIMULATION")
    output_lines.append("=" * 80)
    output_lines.append(f"Workflow Type: {workflow_type}")
    output_lines.append(f"Language: {language}")
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # Load the appropriate prompt library based on language
    language_module_map = {
        "English": "prompt_english",
        "Polish": "prompt_polish",
        "Chinese": "prompt_chinese",
        "Dutch": "prompt_dutch",
        "Greek": "prompt_greek",
    }
    
    # Dynamically import the correct prompt module
    from importlib import import_module
    prompt_module_name = f"core.prompting.{language_module_map.get(language, 'prompt_english')}"
    
    try:
        prompt_module = import_module(prompt_module_name)
        PromptLib = prompt_module.PromptLibrary
        output_lines.append(f"Loaded prompt module: {prompt_module_name}")
    except ImportError as e:
        output_lines.append(f"WARNING: Could not load {prompt_module_name}: {e}")
        output_lines.append("Falling back to English prompts")
        from core import PromptLibrary as PromptLib
    
    output_lines.append("")
    
    # Load data
    data_file = f"prepared_data_{language}/final_batch_judge_fixed.json"
    
    try:
        data = DataLoader.load_from_json(data_file)
    except FileNotFoundError:
        output_lines.append(f"ERROR: Data file not found: {data_file}")
        output_lines.append("Cannot proceed without sample data")
        # Save error message and return
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        return output_file
    
    # Take only requested number of instances
    data = data[:num_instances]
    
    # Get workflow configuration
    wf_config = WORKFLOW_TYPES[workflow_type]
    
    output_lines.append("-" * 80)
    output_lines.append("WORKFLOW CONFIGURATION")
    output_lines.append("-" * 80)
    output_lines.append(f"Name: {wf_config['name']}")
    output_lines.append(f"Description: {wf_config['description']}")
    output_lines.append(f"System Prompt: {wf_config['system_prompt']}")
    output_lines.append(f"Output Format: {wf_config['output_instruction']}")
    if wf_config.get('use_clusters'):
        output_lines.append(f"Uses Clusters: Yes")
    if wf_config.get('use_rubric'):
        output_lines.append(f"Uses Rubric: Yes")
    output_lines.append("")
    
    # Build prompt messages
    if wf_config.get('use_clusters'):
        # For hierarchical/combined, show one cluster as example
        clusters = PromptLib.get_category_clusters()
        cluster_name = list(clusters.keys())[0]  # Get first cluster
        error_ids = clusters[cluster_name]
        
        taxonomy_data = PromptLib.load_taxonomy()
        cluster_taxonomy = {}
        for cat, errors in taxonomy_data.items():
            filtered = [e for e in errors if e.get("id") in error_ids]
            if filtered:
                cluster_taxonomy[cat] = filtered
        
        system_prompt, assignment = build_prompt_messages(workflow_type, PromptLib, cluster_name, cluster_taxonomy)
        output_lines.append(f"NOTE: Showing cluster '{cluster_name}' as example (hierarchical workflows process multiple clusters)")
        output_lines.append("")
    else:
        system_prompt, assignment = build_prompt_messages(workflow_type, PromptLib)
    
    # Process instances
    for idx, instance in enumerate(data, 1):
        instance_id = instance.get("metadata", {}).get("file_id", f"instance_{idx}")
        
        output_lines.append("=" * 80)
        output_lines.append(f"INSTANCE {idx}: {instance_id}")
        output_lines.append("=" * 80)
        output_lines.append("")
        
        # Show instance details
        meta = instance.get("metadata", {})
        code_ctx = instance.get("code_context", {})
        
        output_lines.append("[METADATA]")
        output_lines.append(f"  Language: {meta.get('language', 'unknown')}")
        output_lines.append(f"  File ID:  {instance_id}")
        output_lines.append("")
        
        output_lines.append("[ORIGINAL COMMENT]")
        orig_comment = code_ctx.get("original_comment", "N/A")
        output_lines.append(f"  {orig_comment}")
        output_lines.append("")
        
        output_lines.append("[MODEL PREDICTIONS]")
        for pred in instance.get("model_predictions", [])[:5]:  # Show first 5 models
            model_name = pred.get("model_name", "unknown")
            predicted = pred.get("predicted_comment", "")
            # Show first 100 chars
            preview = predicted[:100] + ('...' if len(predicted) > 100 else '')
            output_lines.append(f"  - {model_name}:")
            output_lines.append(f"    {preview}")
        output_lines.append("")
        
        # Format the instance for the judge
        formatted = PromptLib.format_grouped_instance(instance)
        
        # Display the full prompt
        output_lines.append("#" * 80)
        output_lines.append("# COMPLETE PROMPT SENT TO JUDGE MODEL")
        output_lines.append("#" * 80)
        output_lines.append("")
        
        output_lines.append("=" * 80)
        output_lines.append("[1] SYSTEM MESSAGE")
        output_lines.append("=" * 80)
        output_lines.append(system_prompt)
        output_lines.append("")
        
        output_lines.append("=" * 80)
        output_lines.append("[2] USER MESSAGE")
        output_lines.append("=" * 80)
        output_lines.append(assignment)
        output_lines.append("")
        output_lines.append("-" * 80)
        output_lines.append("--- TEXT TO EVALUATE ---")
        output_lines.append("-" * 80)
        
        # Pretty print the JSON
        try:
            formatted_json = json.loads(formatted)
            output_lines.append(json.dumps(formatted_json, indent=2, ensure_ascii=False))
        except:
            output_lines.append(formatted)
        
        output_lines.append("")
        output_lines.append("=" * 80)
        output_lines.append("")
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    return output_file


def main():
    """Main function to simulate workflows and generate prompt files."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Simulate workflow prompts without making API calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single workflow for English
  python test_prompts_display.py --workflow standard --language English
  
  # Generate all workflows for Polish
  python test_prompts_display.py --language Polish --all-workflows
  
  # Generate all workflows for all languages
  python test_prompts_display.py --all
  
  # Generate specific workflow for 2 instances
  python test_prompts_display.py --workflow cot --language English --num 2
"""
    )
    
    parser.add_argument("--workflow", "-w",
                       choices=list(WORKFLOW_TYPES.keys()),
                       help="Workflow type to simulate")
    parser.add_argument("--language", "-l",
                       help="Language to test (English, Polish, Chinese, Dutch, Greek)")
    parser.add_argument("--num", "-n", type=int, default=1,
                       help="Number of instances to include (default: 1)")
    parser.add_argument("--all-workflows", action="store_true",
                       help="Generate all workflow types for specified language")
    parser.add_argument("--all", action="store_true",
                       help="Generate all workflows for all languages")
    
    args = parser.parse_args()
    
    # Determine what to generate
    if args.all:
        # Generate all workflows for all available languages
        languages = ["English", "Polish", "Dutch", "Chinese", "Greek"]
        workflows = list(WORKFLOW_TYPES.keys())
        
        print(f"\nGenerating {len(workflows)} workflows × {len(languages)} languages = {len(workflows) * len(languages)} files\n")
        
        generated_files = []
        for lang in languages:
            for wf in workflows:
                print(f"Generating: {wf} / {lang}...", end=" ")
                try:
                    output_file = simulate_workflow(wf, lang, args.num)
                    generated_files.append(output_file)
                    print(f"✓ {output_file}")
                except Exception as e:
                    print(f"✗ Error: {e}")
        
        print(f"\nGenerated {len(generated_files)} files:")
        for f in generated_files:
            print(f"  - {f}")
    
    elif args.all_workflows:
        # Generate all workflows for one language
        if not args.language:
            print("ERROR: --language required with --all-workflows")
            sys.exit(1)
        
        workflows = list(WORKFLOW_TYPES.keys())
        print(f"\nGenerating {len(workflows)} workflows for {args.language}\n")
        
        generated_files = []
        for wf in workflows:
            print(f"Generating: {wf}...", end=" ")
            try:
                output_file = simulate_workflow(wf, args.language, args.num)
                generated_files.append(output_file)
                print(f"✓ {output_file}")
            except Exception as e:
                print(f"✗ Error: {e}")
        
        print(f"\nGenerated {len(generated_files)} files:")
        for f in generated_files:
            print(f"  - {f}")
    
    else:
        # Generate single workflow
        if not args.workflow or not args.language:
            print("ERROR: --workflow and --language required (or use --all-workflows / --all)")
            parser.print_help()
            sys.exit(1)
        
        print(f"\nGenerating: {args.workflow} / {args.language}")
        output_file = simulate_workflow(args.workflow, args.language, args.num)
        print(f"✓ Generated: {output_file}\n")


if __name__ == "__main__":
    main()
