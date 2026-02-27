#!/usr/bin/env bash
set -uo pipefail

judges_file="${1:-judges.txt}"
max_concurrent="${2:-1}"
log_dir="logs"

if [[ ! -f "$judges_file" ]]; then
  echo "Judges file not found: $judges_file" >&2
  exit 1
fi
if ! [[ "$max_concurrent" =~ ^[0-9]+$ ]] || [[ "$max_concurrent" -lt 1 ]]; then
  echo "Max concurrent must be a positive integer: $max_concurrent" >&2
  exit 1
fi
mkdir -p "$log_dir"

normalize_lang_dir() {
  local lang
  lang="${1,,}"
  case "$lang" in
    en|english) echo "English" ;;
    zh|chinese) echo "Chinese" ;;
    el|greek) echo "Greek" ;;
    pl|polish) echo "Polish" ;;
    nl|dutch) echo "Dutch" ;;
    *) return 1 ;;
  esac
}

workflow_types=(standard cot hierarchical rubric)

sanitize_name() {
  local raw
  raw="$1"
  echo "$raw" | tr '/:' '__' | tr -cs '[:alnum:]_-' '_'
}

run_workflow_job() {
  local wf_type model prompt_lang data_path ground_path output_dir data_lang log_file
  wf_type="$1"
  model="$2"
  prompt_lang="$3"
  data_lang="$4"
  data_path="$5"
  ground_path="$6"
  output_dir="$7"
  log_file="$8"

  {
    echo "Running type=$wf_type model=$model prompt_lang=$prompt_lang data_lang=$data_lang"
    if ! python run_workflow.py \
      --type "$wf_type" \
      --model "$model" \
      --language "$prompt_lang" \
      --data "$data_path" \
      --ground "$ground_path" \
      --output "$output_dir" \
      --num 100; then
      echo "Workflow failed: type=$wf_type model=$model prompt_lang=$prompt_lang data_lang=$data_lang" >&2
    fi
  } >"$log_file" 2>&1
}

throttle_jobs() {
  local running
  while true; do
    running="$(jobs -rp | wc -l | tr -d ' ')"
    if [[ "$running" -lt "$max_concurrent" ]]; then
      break
    fi
    wait -n || true
  done
}

while IFS=',' read -r model prompt_lang data_lang; do
  # Skip empty lines and comments
  [[ -z "${model// }" ]] && continue
  [[ "${model#\#}" != "$model" ]] && continue

  model="$(echo "$model" | xargs)"
  prompt_lang="$(echo "${prompt_lang:-}" | xargs)"
  data_lang="$(echo "${data_lang:-}" | xargs)"

  if [[ -z "$model" || -z "$prompt_lang" || -z "$data_lang" ]]; then
    echo "Skipping malformed line: $model,$prompt_lang,$data_lang" >&2
    continue
  fi

  if ! data_dir_name="$(normalize_lang_dir "$data_lang")"; then
    echo "Unknown data language: $data_lang" >&2
    exit 1
  fi

  data_path="prepared_data_${data_dir_name}/final_batch_judge.json"
  ground_path="prepared_data_${data_dir_name}/final_batch_pipeline.json"

  model_dir="$(sanitize_name "$model")"
  prompt_lang_dir="$(sanitize_name "$prompt_lang")"
  data_lang_dir="$(sanitize_name "$data_lang")"

  for wf_type in "${workflow_types[@]}"; do
    output_dir="output1/${model_dir}-${prompt_lang_dir}-${data_lang_dir}-${wf_type}"
    log_file="${log_dir}/${model_dir}-${prompt_lang_dir}-${data_lang_dir}-${wf_type}.log"
    echo "Starting type=$wf_type model=$model prompt_lang=$prompt_lang data_lang=$data_lang log=$log_file"
    throttle_jobs
    run_workflow_job "$wf_type" "$model" "$prompt_lang" "$data_lang" "$data_path" "$ground_path" "$output_dir" "$log_file" &
  done

done < "$judges_file"

wait
